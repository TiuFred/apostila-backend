from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import httpx, re, os, json, tempfile, textwrap
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle
from reportlab.graphics import renderPDF
from reportlab.pdfgen import canvas as pdfcanvas
import anthropic

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

api_key = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

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

# ─── Scraping ─────────────────────────────────────────────────────────────────

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
        title = (soup.find("title") or soup.find("h1") or soup.find("h2"))
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

# ─── Claude AI ────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    mode: str
    subject: str
    subject_color: str
    items: List[dict]

def call_claude(prompt: str, system: str = "") -> str:
    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        system=system or "Você é um assistente educacional especialista em criar materiais de estudo didáticos e bem estruturados em português brasileiro.",
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

@app.post("/generate")
async def generate(req: GenerateRequest):
    names = ", ".join(i["title"] for i in req.items)
    contents = []
    for i in req.items:
        c = i.get("scraped_content", "")
        if c:
            contents.append(f"[{i['title']}]: {c[:600]}")
    ctx = "\n\n".join(contents) if contents else ""
    ctx_note = f"\n\nConteúdo extraído dos links:\n{ctx}" if ctx else ""

    prompts = {
        "apostila": f"""Crie uma apostila didática completa sobre "{req.subject}" com base nos autoestudos: {names}.{ctx_note}

Retorne JSON com esta estrutura EXATA:
{{
  "titulo": "Apostila de {req.subject}",
  "introducao": "texto de 2-3 parágrafos introdutórios",
  "secoes": [
    {{
      "titulo": "Título da seção",
      "conteudo": "Conteúdo explicativo detalhado (3-5 parágrafos)",
      "topicos": ["tópico 1", "tópico 2", "tópico 3"],
      "exemplo": "Exemplo prático concreto"
    }}
  ],
  "resumo": "Resumo final em 1 parágrafo",
  "referencias": ["ref 1", "ref 2"]
}}

Retorne SOMENTE o JSON, sem markdown.""",

        "mapa": f"""Crie um mapa mental estruturado sobre "{req.subject}" baseado em: {names}.{ctx_note}

Retorne JSON:
{{
  "centro": "{req.subject}",
  "ramos": [
    {{
      "titulo": "Ramo principal",
      "cor": "#hexcolor",
      "subramos": [
        {{"titulo": "Subramo", "itens": ["item 1", "item 2", "item 3"]}}
      ]
    }}
  ]
}}

Use 4-6 ramos principais. Cores variadas e vibrantes. Retorne SOMENTE o JSON.""",

        "objetiva": f"""Crie 12 questões de múltipla escolha sobre "{req.subject}" baseadas em: {names}.{ctx_note}

Retorne JSON:
{{
  "titulo": "Simulado Objetiva — {req.subject}",
  "questoes": [
    {{
      "numero": 1,
      "enunciado": "Texto da questão",
      "alternativas": {{"a":"texto","b":"texto","c":"texto","d":"texto","e":"texto"}},
      "resposta": "a",
      "justificativa": "Explicação da resposta correta"
    }}
  ]
}}

Varie dificuldade: 4 fáceis, 5 médias, 3 difíceis. Retorne SOMENTE o JSON.""",

        "dissertativa": f"""Crie 6 questões dissertativas sobre "{req.subject}" baseadas em: {names}.{ctx_note}

Retorne JSON:
{{
  "titulo": "Simulado Dissertativo — {req.subject}",
  "questoes": [
    {{
      "numero": 1,
      "enunciado": "Questão aberta desafiadora",
      "valor": "2,0",
      "gabarito": "Resposta completa esperada (3-5 frases)",
      "pontos_chave": ["conceito 1", "conceito 2", "conceito 3"],
      "dificuldade": "Média"
    }}
  ]
}}

Retorne SOMENTE o JSON.""",

        "flashcards": f"""Crie 20 flashcards de revisão sobre "{req.subject}" baseados em: {names}.{ctx_note}

Retorne JSON:
{{
  "titulo": "Flashcards — {req.subject}",
  "cards": [
    {{"id": 1, "frente": "conceito ou pergunta", "verso": "definição ou resposta completa", "categoria": "categoria do card"}}
  ]
}}

Retorne SOMENTE o JSON.""",
    }

    raw = call_claude(prompts[req.mode])
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=500, detail="Erro ao parsear JSON da IA")
    return {"mode": req.mode, "data": data}

# ─── PDF Generation ───────────────────────────────────────────────────────────

class PDFRequest(BaseModel):
    mode: str
    subject: str
    subject_color: str
    data: dict

def make_cover(c_obj, w, h, subject, mode_label, color_hex):
    color = colors.HexColor(color_hex)
    dark = colors.HexColor("#0e0e1a")
    light = colors.HexColor("#ffffff")

    c_obj.setFillColor(dark)
    c_obj.rect(0, 0, w, h, fill=1, stroke=0)

    # Accent stripe
    c_obj.setFillColor(color)
    c_obj.rect(0, h - 8, w, 8, fill=1, stroke=0)
    c_obj.rect(0, 0, 6, h, fill=1, stroke=0)

    # Decorative circles
    c_obj.setFillColorRGB(*[x/255 for x in bytes.fromhex(color_hex.lstrip("#"))], alpha=None)
    c_obj.setFillColor(colors.HexColor(color_hex + "22"))
    c_obj.circle(w - 60, h - 80, 120, fill=1, stroke=0)
    c_obj.setFillColor(colors.HexColor(color_hex + "15"))
    c_obj.circle(w - 40, h - 200, 80, fill=1, stroke=0)

    # Brand
    c_obj.setFillColor(color)
    c_obj.setFont("Helvetica-Bold", 13)
    c_obj.drawString(40, h - 50, "Apostila.ai")

    # Mode label
    c_obj.setFillColor(color)
    c_obj.setFont("Helvetica", 11)
    c_obj.drawString(40, h * 0.62, mode_label.upper())

    # Subject title
    c_obj.setFillColor(light)
    c_obj.setFont("Helvetica-Bold", 36)
    # Wrap long names
    words = subject.split()
    line, lines_out = "", []
    for w2 in words:
        test = (line + " " + w2).strip()
        if c_obj.stringWidth(test, "Helvetica-Bold", 36) > w - 120:
            lines_out.append(line)
            line = w2
        else:
            line = test
    lines_out.append(line)
    y_title = h * 0.55
    for ln in lines_out:
        c_obj.drawString(40, y_title, ln)
        y_title -= 44

    # Divider
    c_obj.setStrokeColor(color)
    c_obj.setLineWidth(1.5)
    c_obj.line(40, y_title - 10, w - 40, y_title - 10)

    # Footer
    c_obj.setFillColor(colors.HexColor("#888888"))
    c_obj.setFont("Helvetica", 9)
    c_obj.drawString(40, 30, "Gerado por Apostila.ai • Material de Estudos")

    c_obj.showPage()

def styles_for(color_hex):
    color = colors.HexColor(color_hex)
    dark = colors.HexColor("#1a1a2e")
    return {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=22, textColor=dark, spaceAfter=6, leading=28),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=15, textColor=color, spaceAfter=4, spaceBefore=14, leading=20),
        "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=12, textColor=dark, spaceAfter=3, spaceBefore=8, leading=16),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, textColor=dark, spaceAfter=6, leading=16),
        "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=10, textColor=dark, spaceAfter=3, leading=14, leftIndent=14, bulletIndent=0),
        "small": ParagraphStyle("small", fontName="Helvetica-Oblique", fontSize=9, textColor=colors.HexColor("#666666"), spaceAfter=4, leading=13),
        "answer": ParagraphStyle("answer", fontName="Helvetica", fontSize=10, textColor=colors.HexColor("#1a6b4a"), spaceAfter=4, leading=14, leftIndent=12),
        "question_num": ParagraphStyle("qnum", fontName="Helvetica-Bold", fontSize=12, textColor=color, spaceAfter=2, leading=16),
    }

def build_apostila_pdf(path, subject, color_hex, data):
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=2.5*cm, rightMargin=2*cm, topMargin=2.5*cm, bottomMargin=2*cm)
    W, H = A4
    c2 = pdfcanvas.Canvas(path, pagesize=A4)
    make_cover(c2, W, H, subject, "Apostila Completa", color_hex)

    st = styles_for(color_hex)
    color = colors.HexColor(color_hex)
    story = []

    story.append(Paragraph(data.get("titulo", f"Apostila de {subject}"), st["title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=color, spaceAfter=12))
    story.append(Paragraph("Introdução", st["h2"]))
    story.append(Paragraph(data.get("introducao", ""), st["body"]))
    story.append(Spacer(1, 10))

    for sec in data.get("secoes", []):
        elems = []
        elems.append(Paragraph(sec["titulo"], st["h2"]))
        elems.append(HRFlowable(width="100%", thickness=0.5, color=color, spaceAfter=6))
        for par in sec["conteudo"].split("\n"):
            if par.strip():
                elems.append(Paragraph(par.strip(), st["body"]))
        if sec.get("topicos"):
            elems.append(Paragraph("Pontos-chave:", st["h3"]))
            for t in sec["topicos"]:
                elems.append(Paragraph(f"• {t}", st["bullet"]))
        if sec.get("exemplo"):
            elems.append(Spacer(1, 6))
            bg = colors.HexColor(color_hex + "18")
            tbl = Table([[Paragraph(f"<b>Exemplo Prático</b><br/>{sec['exemplo']}", st["body"])]], colWidths=[14*cm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), bg),
                ("BOX", (0,0), (-1,-1), 1, color),
                ("LEFTPADDING", (0,0), (-1,-1), 10),
                ("RIGHTPADDING", (0,0), (-1,-1), 10),
                ("TOPPADDING", (0,0), (-1,-1), 8),
                ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                ("ROUNDEDCORNERS", [6]),
            ]))
            elems.append(tbl)
        elems.append(Spacer(1, 8))
        story.append(KeepTogether(elems))

    story.append(PageBreak())
    story.append(Paragraph("Resumo Final", st["h2"]))
    story.append(HRFlowable(width="100%", thickness=1, color=color, spaceAfter=8))
    story.append(Paragraph(data.get("resumo", ""), st["body"]))
    if data.get("referencias"):
        story.append(Spacer(1, 10))
        story.append(Paragraph("Referências", st["h3"]))
        for r in data["referencias"]:
            story.append(Paragraph(f"• {r}", st["bullet"]))

    c2.save()
    # rebuild with content after cover
    import io
    from pypdf import PdfReader, PdfWriter
    buf = io.BytesIO()
    doc2 = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2.5*cm, rightMargin=2*cm, topMargin=2.5*cm, bottomMargin=2*cm)
    doc2.build(story)
    buf.seek(0)
    writer = PdfWriter()
    cover_pdf = PdfReader(path)
    writer.add_page(cover_pdf.pages[0])
    content_pdf = PdfReader(buf)
    for p in content_pdf.pages:
        writer.add_page(p)
    with open(path, "wb") as f:
        writer.write(f)

def build_mapa_pdf(path, subject, color_hex, data):
    from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle, Group
    W, H = A4
    c2 = pdfcanvas.Canvas(path, pagesize=A4)
    make_cover(c2, W, H, subject, "Mapa Mental", color_hex)

    # Mapa mental page
    c2.setFillColor(colors.HexColor("#fafafa"))
    c2.rect(0, 0, W, H, fill=1, stroke=0)

    main_color = colors.HexColor(color_hex)
    cx, cy = W / 2, H / 2 + 20

    # Center node
    c2.setFillColor(main_color)
    c2.roundRect(cx - 80, cy - 22, 160, 44, 22, fill=1, stroke=0)
    c2.setFillColor(colors.white)
    c2.setFont("Helvetica-Bold", 12)
    center_text = data.get("centro", subject)
    tw = c2.stringWidth(center_text, "Helvetica-Bold", 12)
    c2.drawString(cx - tw/2, cy - 4, center_text)

    ramos = data.get("ramos", [])
    import math
    n = len(ramos)
    for i, ramo in enumerate(ramos):
        angle = (2 * math.pi * i / n) - math.pi / 2
        dist = 170
        rx = cx + dist * math.cos(angle)
        ry = cy + dist * math.sin(angle)

        try:
            rc = colors.HexColor(ramo.get("cor", color_hex))
        except:
            rc = main_color

        # Line from center
        c2.setStrokeColor(rc)
        c2.setLineWidth(2)
        c2.line(cx, cy, rx, ry)

        # Branch box
        bw, bh = 120, 30
        c2.setFillColor(rc)
        c2.roundRect(rx - bw/2, ry - bh/2, bw, bh, 8, fill=1, stroke=0)
        c2.setFillColor(colors.white)
        c2.setFont("Helvetica-Bold", 9)
        t = ramo.get("titulo", "")[:18]
        tw = c2.stringWidth(t, "Helvetica-Bold", 9)
        c2.drawString(rx - tw/2, ry - 3, t)

        # Sub-branches
        subs = ramo.get("subramos", [])
        for j, sub in enumerate(subs[:3]):
            sa = angle + (j - len(subs)/2 + 0.5) * 0.45
            sd = 100
            sx = rx + sd * math.cos(sa)
            sy = ry + sd * math.sin(sa)
            c2.setStrokeColor(rc)
            c2.setLineWidth(1)
            c2.line(rx, ry, sx, sy)
            sub_t = sub.get("titulo", "")[:16]
            c2.setFillColor(colors.HexColor("#f0f0f0"))
            sw2 = max(c2.stringWidth(sub_t, "Helvetica", 8) + 14, 70)
            c2.roundRect(sx - sw2/2, sy - 11, sw2, 22, 5, fill=1, stroke=0)
            c2.setFillColor(colors.HexColor("#333333"))
            c2.setFont("Helvetica", 8)
            tw2 = c2.stringWidth(sub_t, "Helvetica", 8)
            c2.drawString(sx - tw2/2, sy - 3, sub_t)

    # Title
    c2.setFillColor(colors.HexColor("#1a1a2e"))
    c2.setFont("Helvetica-Bold", 14)
    c2.drawCentredString(W/2, H - 40, f"Mapa Mental — {subject}")
    c2.setFont("Helvetica", 8)
    c2.setFillColor(colors.HexColor("#999999"))
    c2.drawCentredString(W/2, 20, "Gerado por Apostila.ai")
    c2.showPage()

    # Legend page
    st = styles_for(color_hex)
    story = []
    story.append(Paragraph("Detalhamento do Mapa Mental", st["title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor(color_hex), spaceAfter=12))
    for ramo in ramos:
        try:
            rc = colors.HexColor(ramo.get("cor", color_hex))
        except:
            rc = colors.HexColor(color_hex)
        story.append(Paragraph(ramo.get("titulo",""), ParagraphStyle("rh", fontName="Helvetica-Bold", fontSize=13, textColor=rc, spaceAfter=4, spaceBefore=10)))
        for sub in ramo.get("subramos", []):
            story.append(Paragraph(f"  <b>{sub.get('titulo','')}</b>", st["h3"]))
            for item in sub.get("itens", []):
                story.append(Paragraph(f"    • {item}", st["bullet"]))
        story.append(Spacer(1, 4))

    import io, shutil
    from pypdf import PdfReader, PdfWriter
    # save the canvas (cover + mapa page) first
    c2.save()
    buf = io.BytesIO()
    doc2 = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2.5*cm, rightMargin=2*cm, topMargin=2.5*cm, bottomMargin=2*cm)
    doc2.build(story)
    buf.seek(0)

    writer = PdfWriter()
    cover_reader = PdfReader(path)
    for p in cover_reader.pages:
        writer.add_page(p)
    content_pdf = PdfReader(buf)
    for p in content_pdf.pages:
        writer.add_page(p)
    final = path + "_final.pdf"
    with open(final, "wb") as f:
        writer.write(f)
    shutil.move(final, path)

def build_simulado_pdf(path, subject, color_hex, data, mode):
    W, H = A4
    c2 = pdfcanvas.Canvas(path, pagesize=A4)
    label = "Simulado Objetiva" if mode == "objetiva" else "Simulado Dissertativo"
    make_cover(c2, W, H, subject, label, color_hex)
    c2.save()

    st = styles_for(color_hex)
    color = colors.HexColor(color_hex)
    story = []
    story.append(Paragraph(data.get("titulo", label), st["title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=color, spaceAfter=4))
    total = len(data.get("questoes", []))
    story.append(Paragraph(f"Total: {total} questões  •  Matéria: {subject}", st["small"]))
    story.append(Spacer(1, 12))

    for q in data.get("questoes", []):
        elems = []
        diff = q.get("dificuldade", "")
        diff_color = {"Fácil": "#22C9A0", "Média": "#F7A83E", "Difícil": "#F76A6A"}.get(diff, "#888888")
        header = f"<b>Questão {q['numero']}</b>"
        if diff:
            header += f"  <font color='{diff_color}'>● {diff}</font>"
        if q.get("valor"):
            header += f"  <font color='#888888'>({q['valor']} pts)</font>"
        elems.append(Paragraph(header, st["question_num"]))
        elems.append(Paragraph(q.get("enunciado",""), st["body"]))

        if mode == "objetiva":
            for letra, texto in q.get("alternativas", {}).items():
                elems.append(Paragraph(f"<b>{letra})</b> {texto}", st["bullet"]))
            elems.append(Spacer(1, 4))
            # Gabarito in colored box
            resp = q.get("resposta","")
            just = q.get("justificativa","")
            bg = colors.HexColor(color_hex + "15")
            tbl = Table([[Paragraph(f"<b>Gabarito: {resp.upper()}</b> — {just}", st["answer"])]], colWidths=[14*cm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,-1), bg),
                ("BOX",(0,0),(-1,-1),1,color),
                ("LEFTPADDING",(0,0),(-1,-1),8),
                ("RIGHTPADDING",(0,0),(-1,-1),8),
                ("TOPPADDING",(0,0),(-1,-1),6),
                ("BOTTOMPADDING",(0,0),(-1,-1),6),
            ]))
            elems.append(tbl)
        else:
            elems.append(Spacer(1, 20))
            elems.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=2))
            elems.append(Paragraph("_" * 80, st["small"]))
            elems.append(Paragraph("_" * 80, st["small"]))
            if q.get("gabarito"):
                bg = colors.HexColor(color_hex + "12")
                tbl = Table([[Paragraph(f"<b>Gabarito:</b> {q['gabarito']}", st["answer"])]], colWidths=[14*cm])
                tbl.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,-1),bg),
                    ("BOX",(0,0),(-1,-1),0.5,color),
                    ("LEFTPADDING",(0,0),(-1,-1),8),
                    ("RIGHTPADDING",(0,0),(-1,-1),8),
                    ("TOPPADDING",(0,0),(-1,-1),6),
                    ("BOTTOMPADDING",(0,0),(-1,-1),6),
                ]))
                elems.append(tbl)
            if q.get("pontos_chave"):
                elems.append(Paragraph("Pontos esperados: " + " • ".join(q["pontos_chave"]), st["small"]))

        elems.append(Spacer(1, 10))
        elems.append(HRFlowable(width="100%", thickness=0.3, color=colors.HexColor("#dddddd"), spaceAfter=8))
        story.append(KeepTogether(elems))

    import io
    from pypdf import PdfReader, PdfWriter
    buf = io.BytesIO()
    doc2 = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2.5*cm, rightMargin=2*cm, topMargin=2.5*cm, bottomMargin=2*cm)
    doc2.build(story)
    buf.seek(0)
    writer = PdfWriter()
    cover_pdf = PdfReader(path)
    writer.add_page(cover_pdf.pages[0])
    content_pdf = PdfReader(buf)
    for p in content_pdf.pages:
        writer.add_page(p)
    with open(path, "wb") as f:
        writer.write(f)

def build_flashcards_pdf(path, subject, color_hex, data):
    W, H = A4
    c2 = pdfcanvas.Canvas(path, pagesize=A4)
    make_cover(c2, W, H, subject, "Flashcards de Revisão", color_hex)

    color = colors.HexColor(color_hex)
    dark = colors.HexColor("#1a1a2e")
    cards = data.get("cards", [])
    cards_per_page = 4
    cw, ch = (W - 5*cm) / 2, 6*cm
    margin_x, margin_y = 2*cm, 2.5*cm

    for page_start in range(0, len(cards), cards_per_page):
        page_cards = cards[page_start:page_start + cards_per_page]
        c2.setFillColor(colors.HexColor("#fafafa"))
        c2.rect(0, 0, W, H, fill=1, stroke=0)
        c2.setFillColor(dark)
        c2.setFont("Helvetica-Bold", 13)
        c2.drawString(margin_x, H - 35, f"Flashcards — {subject}")
        c2.setFillColor(color)
        c2.rect(margin_x, H - 42, W - 4*cm, 2, fill=1, stroke=0)

        for idx, card in enumerate(page_cards):
            col = idx % 2
            row = idx // 2
            x = margin_x + col * (cw + 1*cm)
            y = H - 70 - row * (ch + 1*cm) - ch

            # Front
            c2.setFillColor(color)
            c2.roundRect(x, y + ch/2, cw, ch/2 - 2, 6, fill=1, stroke=0)
            c2.setFillColor(colors.white)
            c2.setFont("Helvetica-Bold", 7)
            c2.drawString(x + 8, y + ch - 12, "FRENTE")
            c2.setFont("Helvetica-Bold", 9)
            frente = card.get("frente", "")
            lines = textwrap.wrap(frente, 28)
            yy = y + ch/2 + ch/4 + (len(lines)-1)*6
            for ln in lines[:4]:
                tw = c2.stringWidth(ln, "Helvetica-Bold", 9)
                c2.drawString(x + cw/2 - tw/2, yy, ln)
                yy -= 14

            # Back
            c2.setFillColor(colors.HexColor("#f0f0f8"))
            c2.roundRect(x, y, cw, ch/2 - 2, 6, fill=1, stroke=0)
            c2.setFillColor(colors.HexColor("#666666"))
            c2.setFont("Helvetica-Bold", 7)
            c2.drawString(x + 8, y + ch/2 - 14, "VERSO")
            c2.setFillColor(dark)
            c2.setFont("Helvetica", 8.5)
            verso = card.get("verso", "")
            lines2 = textwrap.wrap(verso, 30)
            yy2 = y + ch/4 + (len(lines2)-1)*5
            for ln in lines2[:4]:
                tw = c2.stringWidth(ln, "Helvetica", 8.5)
                c2.drawString(x + cw/2 - tw/2, yy2, ln)
                yy2 -= 12

            cat = card.get("categoria", "")
            if cat:
                c2.setFillColor(colors.HexColor(color_hex + "30"))
                c2.roundRect(x + 4, y + 4, min(c2.stringWidth(cat, "Helvetica", 7) + 10, 90), 14, 4, fill=1, stroke=0)
                c2.setFillColor(color)
                c2.setFont("Helvetica", 7)
                c2.drawString(x + 9, y + 8, cat[:20])

        c2.setFillColor(colors.HexColor("#aaaaaa"))
        c2.setFont("Helvetica", 8)
        c2.drawCentredString(W/2, 20, f"Apostila.ai • {page_start // cards_per_page + 2} de {(len(cards)-1)//cards_per_page + 2}")
        c2.showPage()

    c2.save()

@app.post("/pdf")
async def make_pdf(req: PDFRequest):
    fname = f"/tmp/apostila_{req.mode}_{req.subject.replace(' ','_')}.pdf"
    try:
        if req.mode == "apostila":
            build_apostila_pdf(fname, req.subject, req.subject_color, req.data)
        elif req.mode == "mapa":
            build_mapa_pdf(fname, req.subject, req.subject_color, req.data)
        elif req.mode in ("objetiva", "dissertativa"):
            build_simulado_pdf(fname, req.subject, req.subject_color, req.data, req.mode)
        elif req.mode == "flashcards":
            build_flashcards_pdf(fname, req.subject, req.subject_color, req.data)
        else:
            raise HTTPException(400, "Modo inválido")
    except Exception as e:
        raise HTTPException(500, str(e))
    return FileResponse(fname, media_type="application/pdf", filename=os.path.basename(fname))

@app.get("/health")
def health():
    return {"ok": True}
