from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.requests import Request
from pydantic import BaseModel
from typing import List, Optional
import httpx, re, os, json, textwrap, base64, io
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas as pdfcanvas
from pypdf import PdfReader, PdfWriter
import anthropic
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
from google.oauth2 import service_account

# ── Google Drive setup ────────────────────────────────────────────────────────

DRIVE_ROOT_FOLDER = "0APH2Y3zPPWOOUk9PVA"
DRIVE_IS_SHARED   = True  # Shared Drive

SUBJECT_SIGLAS = {
    "Programação": "COM",
    "UX":          "UX",
    "Orientação":  "ORI",
    "Liderança":   "LID",
    "Negócios":    "NEG",
    "Matemática":  "MAT",
}

MODE_LABELS = {
    "apostila":     "Apostila",
    "mapa":         "MapaMental",
    "objetiva":     "SimuladoObj",
    "dissertativa": "SimuladoDiss",
    "flashcards":   "Flashcards",
}

def get_drive_service():
    """Build Drive service from env var (JSON string) or file."""
    creds_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if creds_json:
        info = json.loads(creds_json)
    else:
        raise Exception("GOOGLE_SERVICE_ACCOUNT_JSON não configurada")
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

def get_or_create_folder(service, name: str, parent_id: str) -> str:
    """Return folder ID, creating it if it doesn't exist. Supports Shared Drives."""
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false"
    res = service.files().list(
        q=q,
        fields="files(id,name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]
    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = service.files().create(
        body=meta,
        fields="id",
        supportsAllDrives=True
    ).execute()
    return folder["id"]

def upload_to_drive(pdf_path: str, subject: str, weeks: list, mode: str) -> str:
    """Upload PDF to Drive under Subject > Week(s) folder. Returns file URL."""
    service  = get_drive_service()
    sigla    = SUBJECT_SIGLAS.get(subject, subject[:3].upper())
    mode_lbl = MODE_LABELS.get(mode, mode.capitalize())

    # Subject folder
    subject_folder = get_or_create_folder(service, subject, DRIVE_ROOT_FOLDER)

    # Week folder
    if len(weeks) == 1:
        week_name = weeks[0]
    else:
        nums = [w.replace("Semana ", "").strip() for w in sorted(weeks)]
        week_name = f"Semana {'-'.join(nums)}"

    week_folder = get_or_create_folder(service, week_name, subject_folder)

    # File name
    week_num = week_name.replace("Semana ", "").strip()
    file_name = f"{mode_lbl}(S-{week_num}, M-{sigla}).pdf"

    # Read file content
    with open(pdf_path, "rb") as f:
        file_content = f.read()

    # Upload using MediaIoBaseUpload (avoids storage quota issue)
    media = MediaIoBaseUpload(
        io.BytesIO(file_content),
        mimetype="application/pdf",
        resumable=False
    )
    file_meta = {
        "name": file_name,
        "parents": [week_folder]
    }

    uploaded = service.files().create(
        body=file_meta,
        media_body=media,
        fields="id,webViewLink",
        supportsAllDrives=True
    ).execute()

    # Make file readable by anyone with link
    try:
        service.permissions().create(
            fileId=uploaded["id"],
            body={"role": "reader", "type": "anyone"},
            supportsAllDrives=True
        ).execute()
    except Exception:
        pass

    return uploaded.get("webViewLink", f"https://drive.google.com/file/d/{uploaded['id']}/view")


app = FastAPI()
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

def get_client():
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def call_ai(prompt: str) -> str:
    client = get_client()
    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        system="Você é um assistente educacional especialista em criar materiais de estudo didáticos em português brasileiro. Retorne SOMENTE JSON válido, sem markdown, sem texto adicional.",
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True}

# ── Scraping ──────────────────────────────────────────────────────────────────

class ScrapeRequest(BaseModel):
    url: str

@app.post("/scrape")
async def scrape(req: ScrapeRequest):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ApostilaAI/1.0)"}
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
            r = await c.get(req.url, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","nav","footer","header","aside","form","button","iframe"]):
            tag.decompose()
        title = soup.find("title") or soup.find("h1") or soup.find("h2")
        title = title.get_text(strip=True) if title else "Sem título"
        meta_desc = ""
        m = soup.find("meta", attrs={"name": "description"})
        if m: meta_desc = m.get("content", "")
        main = soup.find("main") or soup.find("article") or soup.find("body")
        paragraphs = main.find_all("p") if main else soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 60)
        text = re.sub(r'\s+', ' ', text).strip()
        return {"title": title, "description": meta_desc, "content": text[:4000]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ── PDF Extract (base64) ───────────────────────────────────────────────────────

class ExtractPDFRequest(BaseModel):
    data: str  # base64 encoded PDF

@app.post("/extract-pdf-b64")
async def extract_pdf_b64(req: ExtractPDFRequest):
    try:
        raw = base64.b64decode(req.data)
        reader = PdfReader(io.BytesIO(raw))
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                pages_text.append(f"[Página {i+1}]\n{text}")
        full_text = "\n\n".join(pages_text)
        # Return up to 15000 chars (covers most academic PDFs fully)
        return {"content": full_text[:15000], "pages": len(reader.pages), "chars": len(full_text)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ── Generate ──────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    mode: str
    subject: str
    subject_color: str
    items: List[dict]

@app.post("/generate")
async def generate(req: GenerateRequest):
    # Build rich context from each item
    item_blocks = []
    for i in req.items:
        block = f"### Autoestudo: {i['title']}"
        if i.get("week"):
            block += f" ({i['week']})"
        if i.get("notes"):
            block += f"\nInstruções do usuário: {i['notes']}"
        if i.get("scraped_content"):
            block += f"\nConteúdo extraído:\n{i['scraped_content'][:4000]}"
        elif i.get("url"):
            block += f"\nURL: {i['url']}"
        item_blocks.append(block)

    ctx = "\n\n".join(item_blocks)
    names = ", ".join(i["title"] for i in req.items)
    weeks = sorted(set(i["week"] for i in req.items if i.get("week")))
    week_note = f" (Semanas: {', '.join(weeks)})" if weeks else ""

    base_instruction = f"""Você é um assistente educacional. Crie o material EXCLUSIVAMENTE com base nos conteúdos abaixo dos autoestudos fornecidos. NÃO invente conteúdo genérico. Use apenas o que está presente nos textos extraídos e nas instruções do usuário. Se um autoestudo não tiver conteúdo extraído, use o título como referência temática.

Matéria: {req.subject}{week_note}
Autoestudos selecionados: {names}

{ctx}
"""

    prompts = {
        "apostila": base_instruction + f"""
Agora crie uma apostila didática completa e detalhada baseada EXCLUSIVAMENTE nos conteúdos acima.

Retorne JSON (somente JSON, sem markdown):
{{
  "titulo": "Apostila de {req.subject}{week_note}",
  "introducao": "2-3 parágrafos introdutórios baseados no conteúdo real dos autoestudos",
  "secoes": [
    {{
      "titulo": "Título da seção (derivado do conteúdo real)",
      "conteudo": "Conteúdo explicativo detalhado extraído dos autoestudos (mínimo 3 parágrafos)",
      "topicos": ["tópico concreto 1", "tópico concreto 2", "tópico concreto 3"],
      "exemplo": "Exemplo prático baseado no conteúdo real"
    }}
  ],
  "resumo": "Resumo final baseado no que foi abordado nos autoestudos",
  "referencias": ["título e fonte de cada autoestudo"]
}}""",

        "mapa": base_instruction + f"""
Agora crie um mapa mental estruturado baseado EXCLUSIVAMENTE nos conteúdos acima.

Retorne JSON (somente JSON, sem markdown):
{{
  "centro": "{req.subject}{week_note}",
  "ramos": [
    {{
      "titulo": "Conceito real do conteúdo",
      "cor": "#22C9A0",
      "subramos": [
        {{"titulo": "Subtópico real", "itens": ["detalhe concreto 1", "detalhe concreto 2"]}}
      ]
    }}
  ]
}}
Use 4-6 ramos com cores: #7C6AF7, #22C9A0, #F76A6A, #F7A83E, #4FB8F7, #D46AF7""",

        "objetiva": base_instruction + f"""
Agora crie 12 questões de múltipla escolha baseadas EXCLUSIVAMENTE nos conteúdos acima. As questões devem testar conceitos reais presentes nos textos extraídos.

Retorne JSON (somente JSON, sem markdown):
{{
  "titulo": "Simulado Objetiva — {req.subject}{week_note}",
  "questoes": [
    {{
      "numero": 1,
      "enunciado": "Questão baseada em conceito real do conteúdo",
      "alternativas": {{"a":"texto","b":"texto","c":"texto","d":"texto","e":"texto"}},
      "resposta": "a",
      "justificativa": "Explicação baseada no conteúdo extraído",
      "dificuldade": "Fácil"
    }}
  ]
}}
Varie dificuldade: 4 Fácil, 5 Média, 3 Difícil. O gabarito deve vir por último no PDF.""",

        "dissertativa": base_instruction + f"""
Agora crie 6 questões dissertativas baseadas EXCLUSIVAMENTE nos conteúdos acima. As questões devem exigir reflexão sobre os temas reais dos autoestudos.

Retorne JSON (somente JSON, sem markdown):
{{
  "titulo": "Simulado Dissertativo — {req.subject}{week_note}",
  "questoes": [
    {{
      "numero": 1,
      "enunciado": "Questão aberta sobre conceito real do conteúdo",
      "valor": "2,0",
      "gabarito": "Resposta esperada baseada no conteúdo extraído",
      "pontos_chave": ["conceito real 1", "conceito real 2", "conceito real 3"],
      "dificuldade": "Média"
    }}
  ]
}}""",

        "flashcards": base_instruction + f"""
Agora crie 20 flashcards de revisão baseados EXCLUSIVAMENTE nos conteúdos acima. Cada card deve conter um conceito real presente nos textos.

Retorne JSON (somente JSON, sem markdown):
{{
  "titulo": "Flashcards — {req.subject}{week_note}",
  "cards": [
    {{"id": 1, "frente": "Conceito ou termo real do conteúdo", "verso": "Definição ou explicação baseada no texto extraído", "categoria": "categoria do autoestudo"}}
  ]
}}""",
    }

    if req.mode not in prompts:
        raise HTTPException(status_code=400, detail="Modo inválido")

    raw = call_ai(prompts[req.mode])
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=500, detail=f"Erro ao parsear JSON: {raw[:200]}")
    return {"mode": req.mode, "data": data}

# ── PDF Generation ────────────────────────────────────────────────────────────

class PDFRequest(BaseModel):
    mode: str
    subject: str
    subject_color: str
    data: dict

SUBJECT_COLORS = {
    "#7C6AF7": colors.HexColor("#7C6AF7"),
    "#22C9A0": colors.HexColor("#22C9A0"),
    "#F76A6A": colors.HexColor("#F76A6A"),
    "#F7A83E": colors.HexColor("#F7A83E"),
    "#4FB8F7": colors.HexColor("#4FB8F7"),
    "#D46AF7": colors.HexColor("#D46AF7"),
    "#6AF7A8": colors.HexColor("#6AF7A8"),
    "#F7D46A": colors.HexColor("#F7D46A"),
}

def get_color(hex_str):
    return SUBJECT_COLORS.get(hex_str, colors.HexColor("#7C6AF7"))

def make_cover(c_obj, w, h, subject, mode_label, color_hex):
    color = colors.HexColor(color_hex)
    dark  = colors.HexColor("#0e0e1a")
    c_obj.setFillColor(dark); c_obj.rect(0,0,w,h,fill=1,stroke=0)
    c_obj.setFillColor(color); c_obj.rect(0,h-8,w,8,fill=1,stroke=0); c_obj.rect(0,0,6,h,fill=1,stroke=0)
    c_obj.setFillColor(colors.HexColor(color_hex+"22")); c_obj.circle(w-60,h-80,120,fill=1,stroke=0)
    c_obj.setFillColor(colors.HexColor(color_hex+"15")); c_obj.circle(w-40,h-200,80,fill=1,stroke=0)
    c_obj.setFillColor(color); c_obj.setFont("Helvetica-Bold",13); c_obj.drawString(40,h-50,"Apostila.ai")
    c_obj.setFillColor(color); c_obj.setFont("Helvetica",11); c_obj.drawString(40,h*0.62,mode_label.upper())
    c_obj.setFillColor(colors.white); c_obj.setFont("Helvetica-Bold",36)
    words = subject.split(); line = ""; lines_out = []
    for wd in words:
        test = (line+" "+wd).strip()
        if c_obj.stringWidth(test,"Helvetica-Bold",36) > w-120: lines_out.append(line); line=wd
        else: line=test
    lines_out.append(line)
    y_title = h*0.55
    for ln in lines_out: c_obj.drawString(40,y_title,ln); y_title-=44
    c_obj.setStrokeColor(color); c_obj.setLineWidth(1.5); c_obj.line(40,y_title-10,w-40,y_title-10)
    c_obj.setFillColor(colors.HexColor("#888888")); c_obj.setFont("Helvetica",9)
    c_obj.drawString(40,30,"Gerado por Apostila.ai • Material de Estudos")
    c_obj.showPage()

def sty(name,**kw):
    d=dict(fontName="Helvetica",fontSize=10.5,textColor=colors.HexColor("#1a1a2e"),leading=16,spaceAfter=7)
    d.update(kw); return ParagraphStyle(name,**d)

def build_apostila_pdf(path, subject, color_hex, data):
    color = get_color(color_hex)
    dark  = colors.HexColor("#1a1a2e")
    ST = {
        "title": sty("t",fontName="Helvetica-Bold",fontSize=22,spaceAfter=6,leading=28),
        "h2":    sty("h2",fontName="Helvetica-Bold",fontSize=15,textColor=color,spaceBefore=14,spaceAfter=4,leading=20),
        "h3":    sty("h3",fontName="Helvetica-Bold",fontSize=12,spaceBefore=8,spaceAfter=3,leading=16),
        "body":  sty("body"),
        "bullet":sty("bul",leftIndent=14,spaceAfter=3,leading=14),
        "small": sty("sm",fontName="Helvetica-Oblique",fontSize=9,textColor=colors.HexColor("#666666"),spaceAfter=4,leading=13),
        "ok":    sty("ok",fontSize=10,textColor=colors.HexColor("#1a6b4a"),spaceAfter=3,leading=14),
    }
    # cover
    cv = pdfcanvas.Canvas(path, pagesize=A4); W,H=A4
    make_cover(cv, W, H, subject, "Apostila Completa", color_hex); cv.save()
    # content
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2.5*cm, rightMargin=2*cm, topMargin=2.5*cm, bottomMargin=2*cm)
    story = []
    story += [Paragraph(data.get("titulo", f"Apostila de {subject}"), ST["title"]),
              HRFlowable(width="100%",thickness=2,color=color,spaceAfter=12)]
    story += [Paragraph("Introdução", ST["h2"]), Paragraph(data.get("introducao",""), ST["body"]), Spacer(1,10)]
    for sec in data.get("secoes",[]):
        elems = [Paragraph(sec["titulo"], ST["h2"]), HRFlowable(width="100%",thickness=0.5,color=color,spaceAfter=6)]
        for par in sec.get("conteudo","").split("\n"):
            if par.strip(): elems.append(Paragraph(par.strip(), ST["body"]))
        if sec.get("topicos"):
            elems.append(Paragraph("Pontos-chave:", ST["h3"]))
            for t in sec["topicos"]: elems.append(Paragraph(f"• {t}", ST["bullet"]))
        if sec.get("exemplo"):
            bg = colors.HexColor(color_hex+"18")
            tbl = Table([[Paragraph(f"<b>Exemplo:</b> {sec['exemplo']}", ST["body"])]], colWidths=[14*cm])
            tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),bg),("BOX",(0,0),(-1,-1),1,color),
                ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
                ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8)]))
            elems.append(tbl)
        elems.append(Spacer(1,8)); story.append(KeepTogether(elems))
    story += [PageBreak(), Paragraph("Resumo Final", ST["h2"]),
              HRFlowable(width="100%",thickness=1,color=color,spaceAfter=8),
              Paragraph(data.get("resumo",""), ST["body"])]
    if data.get("referencias"):
        story += [Spacer(1,10), Paragraph("Referências", ST["h3"])]
        for r in data["referencias"]: story.append(Paragraph(f"• {r}", ST["bullet"]))
    doc.build(story); buf.seek(0)
    w = PdfWriter()
    for p in PdfReader(path).pages: w.add_page(p)
    for p in PdfReader(buf).pages: w.add_page(p)
    with open(path,"wb") as f: w.write(f)

def build_mapa_pdf(path, subject, color_hex, data):
    import math
    W,H = A4
    cv = pdfcanvas.Canvas(path, pagesize=A4)
    make_cover(cv, W, H, subject, "Mapa Mental", color_hex)
    # mapa page
    cv.setFillColor(colors.HexColor("#fafafa")); cv.rect(0,0,W,H,fill=1,stroke=0)
    mc = colors.HexColor(color_hex)
    cx,cy = W/2, H/2+20
    cv.setFillColor(mc); cv.roundRect(cx-80,cy-22,160,44,22,fill=1,stroke=0)
    cv.setFillColor(colors.white); cv.setFont("Helvetica-Bold",12)
    ct = data.get("centro",subject); tw=cv.stringWidth(ct,"Helvetica-Bold",12)
    cv.drawString(cx-tw/2,cy-4,ct)
    ramos = data.get("ramos",[])
    n = len(ramos)
    for i,ramo in enumerate(ramos):
        angle=(2*math.pi*i/n)-math.pi/2
        rx=cx+170*math.cos(angle); ry=cy+170*math.sin(angle)
        try: rc=colors.HexColor(ramo.get("cor",color_hex))
        except: rc=mc
        cv.setStrokeColor(rc); cv.setLineWidth(2); cv.line(cx,cy,rx,ry)
        cv.setFillColor(rc); cv.roundRect(rx-60,ry-15,120,30,8,fill=1,stroke=0)
        cv.setFillColor(colors.white); cv.setFont("Helvetica-Bold",9)
        t=ramo.get("titulo","")[:18]; tw=cv.stringWidth(t,"Helvetica-Bold",9)
        cv.drawString(rx-tw/2,ry-3,t)
        for j,sub in enumerate(ramo.get("subramos",[])[:3]):
            sa=angle+(j-len(ramo.get("subramos",[])[:3])/2+0.5)*0.45
            sx=rx+100*math.cos(sa); sy=ry+100*math.sin(sa)
            cv.setStrokeColor(rc); cv.setLineWidth(1); cv.line(rx,ry,sx,sy)
            st2=sub.get("titulo","")[:16]
            sw=max(cv.stringWidth(st2,"Helvetica",8)+14,70)
            cv.setFillColor(colors.HexColor("#f0f0f0")); cv.roundRect(sx-sw/2,sy-11,sw,22,5,fill=1,stroke=0)
            cv.setFillColor(colors.HexColor("#333333")); cv.setFont("Helvetica",8)
            tw2=cv.stringWidth(st2,"Helvetica",8); cv.drawString(sx-tw2/2,sy-3,st2)
    cv.setFillColor(colors.HexColor("#1a1a2e")); cv.setFont("Helvetica-Bold",14)
    cv.drawCentredString(W/2,H-40,f"Mapa Mental — {subject}")
    cv.setFont("Helvetica",8); cv.setFillColor(colors.HexColor("#999999"))
    cv.drawCentredString(W/2,20,"Gerado por Apostila.ai")
    cv.showPage()
    # detail page
    color_rl = get_color(color_hex)
    ST2 = {"title":sty("t2",fontName="Helvetica-Bold",fontSize=18,spaceAfter=6,leading=24),
           "body":sty("b2"), "h3":sty("h3b",fontName="Helvetica-Bold",fontSize=12,spaceAfter=3,leading=16),
           "bullet":sty("bul2",leftIndent=14,spaceAfter=3,leading=14)}
    story=[Paragraph("Detalhamento do Mapa Mental",ST2["title"]),
           HRFlowable(width="100%",thickness=2,color=color_rl,spaceAfter=12)]
    for ramo in ramos:
        try: rc2=colors.HexColor(ramo.get("cor",color_hex))
        except: rc2=color_rl
        story.append(Paragraph(ramo.get("titulo",""),ParagraphStyle("rh",fontName="Helvetica-Bold",fontSize=13,textColor=rc2,spaceAfter=4,spaceBefore=10)))
        for sub in ramo.get("subramos",[]):
            story.append(Paragraph(f"  <b>{sub.get('titulo','')}</b>",ST2["h3"]))
            for item in sub.get("itens",[]): story.append(Paragraph(f"    • {item}",ST2["bullet"]))
        story.append(Spacer(1,4))
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=A4,leftMargin=2.5*cm,rightMargin=2*cm,topMargin=2.5*cm,bottomMargin=2*cm)
    doc.build(story); buf.seek(0)
    w=PdfWriter()
    for p in PdfReader(path).pages: w.add_page(p)
    for p in PdfReader(buf).pages: w.add_page(p)
    import shutil; tmp=path+"_tmp.pdf"
    with open(tmp,"wb") as f: w.write(f)
    shutil.move(tmp,path)

def build_simulado_pdf(path, subject, color_hex, data, mode):
    color = get_color(color_hex)
    label = "Simulado Objetiva" if mode=="objetiva" else "Simulado Dissertativo"
    cv = pdfcanvas.Canvas(path, pagesize=A4); W,H=A4
    make_cover(cv, W, H, subject, label, color_hex); cv.save()
    ST = {
        "title":  sty("t",fontName="Helvetica-Bold",fontSize=22,spaceAfter=6,leading=28),
        "small":  sty("sm",fontName="Helvetica-Oblique",fontSize=9,textColor=colors.HexColor("#666666"),spaceAfter=4,leading=13),
        "qnum":   sty("qn",fontName="Helvetica-Bold",fontSize=12,textColor=color,spaceAfter=2,leading=16),
        "body":   sty("body"),
        "bullet": sty("bul",leftIndent=14,spaceAfter=3,leading=14),
        "answer": sty("ans",fontSize=10,textColor=colors.HexColor("#1a6b4a"),spaceAfter=4,leading=14,leftIndent=12),
    }
    story=[Paragraph(data.get("titulo",label),ST["title"]),
           HRFlowable(width="100%",thickness=2,color=color,spaceAfter=4),
           Paragraph(f"Total: {len(data.get('questoes',[]))} questões  •  {subject}",ST["small"]),
           Spacer(1,12)]
    for q in data.get("questoes",[]):
        elems=[]
        diff=q.get("dificuldade","")
        dc={"Fácil":"#22C9A0","Média":"#F7A83E","Difícil":"#F76A6A"}.get(diff,"#888888")
        hdr=f"<b>Questão {q['numero']}</b>"
        if diff: hdr+=f"  <font color='{dc}'>● {diff}</font>"
        if q.get("valor"): hdr+=f"  <font color='#888888'>({q['valor']} pts)</font>"
        elems.append(Paragraph(hdr,ST["qnum"]))
        elems.append(Paragraph(q.get("enunciado",""),ST["body"]))
        if mode=="objetiva":
            for letra,texto in q.get("alternativas",{}).items():
                elems.append(Paragraph(f"<b>{letra})</b> {texto}",ST["bullet"]))
            elems.append(Spacer(1,4))
            resp=q.get("resposta",""); just=q.get("justificativa","")
            bg=colors.HexColor(color_hex+"15")
            tbl=Table([[Paragraph(f"<b>Gabarito: {resp.upper()}</b> — {just}",ST["answer"])]],colWidths=[14*cm])
            tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),bg),("BOX",(0,0),(-1,-1),1,color),
                ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
                ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
            elems.append(tbl)
        else:
            elems+=[Spacer(1,20),HRFlowable(width="100%",thickness=0.5,color=colors.HexColor("#cccccc"),spaceAfter=2)]
            if q.get("gabarito"):
                bg=colors.HexColor(color_hex+"12")
                tbl=Table([[Paragraph(f"<b>Gabarito:</b> {q['gabarito']}",ST["answer"])]],colWidths=[14*cm])
                tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),bg),("BOX",(0,0),(-1,-1),0.5,color),
                    ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
                    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
                elems.append(tbl)
            if q.get("pontos_chave"):
                elems.append(Paragraph("Pontos: "+" • ".join(q["pontos_chave"]),ST["small"]))
        elems+=[Spacer(1,10),HRFlowable(width="100%",thickness=0.3,color=colors.HexColor("#dddddd"),spaceAfter=8)]
        story.append(KeepTogether(elems))
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=A4,leftMargin=2.5*cm,rightMargin=2*cm,topMargin=2.5*cm,bottomMargin=2*cm)
    doc.build(story); buf.seek(0)
    w=PdfWriter()
    for p in PdfReader(path).pages: w.add_page(p)
    for p in PdfReader(buf).pages: w.add_page(p)
    with open(path,"wb") as f: w.write(f)

def build_flashcards_pdf(path, subject, color_hex, data):
    W,H=A4
    color=get_color(color_hex); dark=colors.HexColor("#1a1a2e")
    cv=pdfcanvas.Canvas(path,pagesize=A4)
    make_cover(cv,W,H,subject,"Flashcards de Revisão",color_hex)
    cards=data.get("cards",[]); cw,ch=(W-5*cm)/2,6*cm; mx,my=2*cm,2.5*cm
    for ps in range(0,len(cards),4):
        pc=cards[ps:ps+4]
        cv.setFillColor(colors.HexColor("#fafafa")); cv.rect(0,0,W,H,fill=1,stroke=0)
        cv.setFillColor(dark); cv.setFont("Helvetica-Bold",13); cv.drawString(mx,H-35,f"Flashcards — {subject}")
        cv.setFillColor(color); cv.rect(mx,H-42,W-4*cm,2,fill=1,stroke=0)
        for idx,card in enumerate(pc):
            col=idx%2; row=idx//2
            x=mx+col*(cw+1*cm); y=H-70-row*(ch+1*cm)-ch
            cv.setFillColor(color); cv.roundRect(x,y+ch/2,cw,ch/2-2,6,fill=1,stroke=0)
            cv.setFillColor(colors.white); cv.setFont("Helvetica-Bold",7); cv.drawString(x+8,y+ch-12,"FRENTE")
            cv.setFont("Helvetica-Bold",9)
            lines=textwrap.wrap(card.get("frente",""),28)
            yy=y+ch/2+ch/4+(len(lines)-1)*6
            for ln in lines[:4]:
                tw=cv.stringWidth(ln,"Helvetica-Bold",9); cv.drawString(x+cw/2-tw/2,yy,ln); yy-=14
            cv.setFillColor(colors.HexColor("#f0f0f8")); cv.roundRect(x,y,cw,ch/2-2,6,fill=1,stroke=0)
            cv.setFillColor(colors.HexColor("#666666")); cv.setFont("Helvetica-Bold",7); cv.drawString(x+8,y+ch/2-14,"VERSO")
            cv.setFillColor(dark); cv.setFont("Helvetica",8.5)
            lines2=textwrap.wrap(card.get("verso",""),30)
            yy2=y+ch/4+(len(lines2)-1)*5
            for ln in lines2[:4]:
                tw=cv.stringWidth(ln,"Helvetica",8.5); cv.drawString(x+cw/2-tw/2,yy2,ln); yy2-=12
            cat=card.get("categoria","")
            if cat:
                cv.setFillColor(colors.HexColor(color_hex+"30"))
                cv.roundRect(x+4,y+4,min(cv.stringWidth(cat,"Helvetica",7)+10,90),14,4,fill=1,stroke=0)
                cv.setFillColor(color); cv.setFont("Helvetica",7); cv.drawString(x+9,y+8,cat[:20])
        cv.setFillColor(colors.HexColor("#aaaaaa")); cv.setFont("Helvetica",8)
        cv.drawCentredString(W/2,20,f"Apostila.ai • {ps//4+2} de {(len(cards)-1)//4+2}")
        cv.showPage()
    cv.save()

@app.post("/pdf")
async def make_pdf(req: PDFRequest):
    fname = f"/tmp/apostila_{req.mode}_{req.subject.replace(' ','_')}.pdf"
    try:
        if req.mode == "apostila":
            build_apostila_pdf(fname, req.subject, req.subject_color, req.data)
        elif req.mode == "mapa":
            build_mapa_pdf(fname, req.subject, req.subject_color, req.data)
        elif req.mode in ("objetiva","dissertativa"):
            build_simulado_pdf(fname, req.subject, req.subject_color, req.data, req.mode)
        elif req.mode == "flashcards":
            build_flashcards_pdf(fname, req.subject, req.subject_color, req.data)
        else:
            raise HTTPException(400, "Modo inválido")
    except Exception as e:
        raise HTTPException(500, str(e))
    return FileResponse(fname, media_type="application/pdf", filename=os.path.basename(fname))

# ── Google Drive Upload ────────────────────────────────────────────────────────

class DriveUploadRequest(BaseModel):
    mode: str
    subject: str
    subject_color: str
    weeks: List[str]
    data: dict

@app.post("/upload-drive")
async def upload_drive(req: DriveUploadRequest):
    # 1. Generate PDF first
    fname = f"/tmp/drive_{req.mode}_{req.subject.replace(' ','_')}.pdf"
    try:
        if req.mode == "apostila":
            build_apostila_pdf(fname, req.subject, req.subject_color, req.data)
        elif req.mode == "mapa":
            build_mapa_pdf(fname, req.subject, req.subject_color, req.data)
        elif req.mode in ("objetiva","dissertativa"):
            build_simulado_pdf(fname, req.subject, req.subject_color, req.data, req.mode)
        elif req.mode == "flashcards":
            build_flashcards_pdf(fname, req.subject, req.subject_color, req.data)
        else:
            raise HTTPException(400, "Modo inválido")
    except Exception as e:
        raise HTTPException(500, f"Erro ao gerar PDF: {e}")

    # 2. Upload to Drive
    try:
        link = upload_to_drive(fname, req.subject, req.weeks, req.mode)
    except Exception as e:
        raise HTTPException(500, f"Erro ao enviar para o Drive: {e}")

    return {"link": link, "message": "Arquivo enviado com sucesso!"}
