# Apostila.ai — Backend

API REST do **Apostila.ai**, responsável por web scraping, extração de PDFs, geração de materiais com IA (Claude), geração de PDFs formatados e integração com Google Drive.

## Stack

- **FastAPI** + Uvicorn
- **Anthropic Claude Opus** — geração de materiais
- **ReportLab** — geração de PDFs
- **Google Drive API** — upload de arquivos
- **BeautifulSoup4** — web scraping
- Deploy: **Railway**

## Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/health` | Status do servidor |
| `GET` | `/credits` | Uso acumulado de tokens (via Supabase) |
| `POST` | `/scrape` | Extrai texto de uma URL |
| `POST` | `/extract-pdf-b64` | Extrai texto de PDF em base64 (até 50.000 chars) |
| `POST` | `/generate` | Gera material com IA (apostila, mapa, simulado, flashcards, desespero) |
| `POST` | `/pdf` | Gera PDF formatado de um material gerado |
| `POST` | `/upload-drive` | Gera PDF e envia para o Google Drive compartilhado |
| `POST` | `/upload-pdf-drive` | Envia PDF de autoestudo original para o Drive |

## Modos de geração (`/generate`)

```json
{
  "mode": "apostila | mapa | objetiva | dissertativa | flashcards | desespero",
  "subject": "Programação",
  "subject_color": "#7C6AF7",
  "items": [
    {
      "title": "Aula 1",
      "week": "Semana 01",
      "notes": "Foque nas páginas 10-25",
      "scraped_content": "texto extraído..."
    }
  ]
}
```

A IA usa o conteúdo extraído como base e **amplia com conhecimento próprio** para gerar materiais ricos e didáticos.

## Estrutura do Google Drive

Os arquivos são organizados automaticamente:

```
Apostila.ai/                          ← pasta raiz (Shared Drive)
├── Programação/
│   ├── Semana 01/
│   │   ├── Autoestudo/
│   │   │   └── nome-do-autoestudo.pdf
│   │   └── Apostila(S-01, M-COM).pdf
│   └── Semana 02/
│       └── SimuladoObj(S-02, M-COM).pdf
└── UX/
    └── Semana 01/
        └── Flashcards(S-01, M-UX).pdf
```

Siglas por matéria: `COM` (Programação), `UX`, `ORI` (Orientação), `LID` (Liderança), `NEG` (Negócios), `MAT` (Matemática)

## Configuração

### Variáveis de ambiente (Railway)

```env
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
```

### Instalação local

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### requirements.txt

```
fastapi==0.115.0
uvicorn==0.30.6
httpx==0.27.0
beautifulsoup4==4.12.3
reportlab==4.2.2
pypdf==4.3.1
anthropic==0.34.2
python-multipart==0.0.9
google-api-python-client==2.140.0
google-auth==2.35.0
```

## Google Drive — configuração da conta de serviço

1. Crie um projeto no [Google Cloud Console](https://console.cloud.google.com)
2. Ative a **Google Drive API**
3. Crie uma **Service Account** e baixe o JSON de credenciais
4. No Google Drive, compartilhe o **Shared Drive** com o e-mail da service account como **Gerenciador de conteúdo**
5. Cole o JSON completo na variável `GOOGLE_SERVICE_ACCOUNT_JSON` no Railway

> O Shared Drive é obrigatório — contas de serviço não têm cota de armazenamento em Drives pessoais.

## Rastreamento de uso

Cada chamada à IA registra automaticamente na tabela `usage_log` do Supabase:

```sql
create table usage_log (
  id uuid default gen_random_uuid() primary key,
  mode text,
  subject text,
  input_tokens int,
  output_tokens int,
  cost_usd numeric(10,6),
  created_at timestamp default now()
);
```

Preço utilizado para cálculo: **Claude Opus** — $15/M tokens de entrada, $75/M tokens de saída.

## Deploy (Railway)

1. Conecte o repositório ao Railway
2. Configure todas as variáveis de ambiente
3. O deploy é automático a cada push na branch `main`
4. A porta é configurada automaticamente via variável `PORT`

## Estrutura de arquivos

```
apostila-backend/
├── main.py              # Aplicação completa
├── requirements.txt
└── Procfile             # (opcional) web: uvicorn main:app --host 0.0.0.0 --port $PORT
```
