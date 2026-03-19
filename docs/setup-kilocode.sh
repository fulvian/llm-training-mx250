#!/bin/bash

# Kilocode CLI Setup Script
# Installa e configura l'ambiente di sviluppo agentico

# Colori
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Funzioni helper
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]{NC} $1"
}
print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "Comando '$1' non trovato"
        exit 1
    fi
}

# Header
echo "=================================================="
echo "   Kilocode CLI Setup Script"
echo "   Ottimizzazione Flusso di Lavoro Agentico"
echo "=================================================="
echo ""

# Check prerequisiti
echo "Controllo prerequisiti..."
check_command "node"
check_command "npm"

check_command "npx"

# 1. Installa agentget
echo ""
echo "1. Installazione agentget..."
if ! npm list -g agentget &> /dev/null 2>&1; then
    print_status "agentget già installato"
else
    npm install -g agentget
    print_status "agentget installato"
fi

echo ""

# 2. Installa superpowers skills
echo "2. Installazione skills da superpowers..."
npx agentget add obra/superpowers --all --skills-only
print_status "Skills superpowers installati"
echo ""

# 3. Installa agenti specializzati
echo "3. Installazione agenti specializzati..."
npx agentget add joeyism/agentget --agent atlas
npx agentget add joeyism/agentget --agent oracle
npx agentget add joeyism/agentget --agent explore
print_status "Agenti specializzati installati"
echo ""

# 4. Backup MCP config esistente
echo "4. Backup configurazione MCP esistente..."
if [ -f ~/.kilocode/mcp.json ]; then
    cp ~/.kilocode/mcp.json ~/.kilocode/mcp.json.backup
    print_status "Backup creato"
fi
echo ""

# 5. Verifica installazione
echo "5. Verifica installazione..."
echo ""
echo "Skills installati:"
npx agentget list --skills-only
echo ""
echo "Agenti installati:"
npx agentget list --agents-only
echo ""

# 6. Configurazione ambiente
echo "6. Configurazione ambiente..."
echo ""
echo "Aggiungi questi alias al tuo shell config (~/.bashrc o ~/.zshrc):"
echo ""
echo "# Kilocode aliases" >> ~/.bashrc
echo "alias kc-skills='npx agentget list --skills-only'" >> ~/.bashrc
echo "alias kc-agents='npx agentget list --agents-only'" >> ~/.bashrc
echo "alias kc-plan='cat task_plan.md 2>/dev/null || echo \"No plan found\"'" >> ~/.bashrc
echo "alias kc-init='bash /path/to/setup-kilocode.sh'" >> ~/.bashrc
echo ""
echo "Alias aggiunti. Ricarica la shell per attivarli."
echo ""

# 7. Installa MCP servers aggiuntivi (opzionale)
echo "7. MCP servers aggiuntivi (opzionale)..."
echo ""
echo "Per installare Puppeteer MCP server (per testing):"
echo "  npm install -g @modelcontextprotocol/server-puppeteer"
echo ""
echo "Per installare SQLite MCP server (per database):"
echo "  npm install -g @modelcontextprotocol/server-sqlite"
echo ""

# Completamento
echo ""
echo "=================================================="
echo "   SETUP COMPLETATO!"
echo "=================================================="
echo ""
echo "Prossimi passi:"
echo "1. Riavvia la shell per attivare gli alias"
echo "2. Verifica le skills con: kc-skills"
echo "3. Verifica gli agenti con: kc-agents"
echo "4. Crea un piano con: kc-plan"
echo ""
echo "Documentazione completa in: docs/kilocode-optimization-plan.md"
