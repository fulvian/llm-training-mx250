# Piano di Ottimizzazione Flusso di Lavoro Agentico su Kilocode CLI

## Executive Summary

Questo documento definisce un piano strategico per migliorare e ottimizzare il flusso di lavoro agentico su Kilocode CLI, basato sull'analisi di:
- **agentget** (joeyism/agentget) - Framework per installazione agenti/skills
- **superpowers** (obra/superpowers) - Metodologia di sviluppo agentic
- **skills.sh** - Ecosistema aperto per skills
- **Configurazione attuale** - Setup esistente in ~/.kilocode/

---

## 1. Analisi dello Stato Attuale

### 1.1 Configurazione Kilocode Esistente

| Componente | Stato Attuale | Note |
|------------|---------------|------|
| MCP Servers | 6 attivi | filesystem, git, github, fetch, memory, sequential-thinking |
| Skills | 1 (planning-with-files) | Basato su principi Manus, con hooks e templates |
| Plugin | 7.0.49 | Versione recente |
| Struttura | Base | Manca organizzazione modulare |

### 1.2 Gap Identificati

| Gap | Impatto | Priorità |
|-----|---------|----------|
| Mancanza di skills per TDD | Alto | Alta |
| Nessun sistema di code review | Alto | Alta |
| Workflow debugging non strutturato | Medio | Alta |
| Assenza di agent specializzati | Alto | Media |
| MCP servers limitati | Medio | Media |
| Nessuna integrazione con skills.sh | Basso | Bassa |

---

## 2. Architettura Target

### 2.1 Struttura Directory Proposta

```
~/.kilocode/
├── mcp.json                    # Configurazione MCP ottimizzata
├── package.json                # Dipendenze plugin
├── instructions/               # Istruzioni globali
│   └── development.instructions.md
├── rules/                      # Regole globali
│   ├── code-quality.rules.md
│   └── security.rules.md
├── agents/                     # Agenti specializzati
│   ├── orchestrator.agent.md
│   ├── code-reviewer.agent.md
│   ├── debugger.agent.md
│   ├── researcher.agent.md
│   └── tester.agent.md
└── skills/                     # Skills modulari
    ├── planning-with-files/    # Esistente
    ├── test-driven-development/
    ├── systematic-debugging/
    ├── code-review/
    ├── git-workflow/
    └── brainstorming/
```

### 2.2 Skills da Implementare (Priorità)

#### Wave 1 - Fondamentali (Alta Priorità)

| Skill | Fonte | Descrizione |
|-------|-------|-------------|
| `test-driven-development` | superpowers | Ciclo RED-GREEN-REFACTOR |
| `systematic-debugging` | superpowers | Processo root cause in 4 fasi |
| `code-review` | custom + superpowers | Review strutturata con checklist |
| `brainstorming` | superpowers | Raffinamento design Socratico |

#### Wave 2 - Workflow (Media Priorità)

| Skill | Fonte | Descrizione |
|-------|-------|-------------|
| `writing-plans` | superpowers | Piani di implementazione dettagliati |
| `executing-plans` | superpowers | Esecuzione batch con checkpoints |
| `git-workflow` | custom + superpowers | Branch, commit, PR automation |
| `subagent-driven-development` | superpowers | Sviluppo con subagent paralleli |

#### Wave 3 - Avanzate (Bassa Priorità)

| Skill | Fonte | Descrizione |
|-------|-------|-------------|
| `dispatching-parallel-agents` | superpowers | Workflow subagent concorrenti |
| `verification-before-completion` | superpowers | Verifica pre-completamento |
| `finishing-a-development-branch` | superpowers | Workflow merge/PR |

---

## 3. Configurazione MCP Ottimizzata

### 3.1 MCP Servers Aggiuntivi Raccomandati

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/fulvio/coding"],
      "disabled": false
    },
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git"],
      "disabled": false
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      },
      "disabled": false
    },
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"],
      "disabled": false
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "disabled": false
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
      "disabled": false
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "disabled": true,
      "note": "Abilitare solo per progetti con DB"
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "${BRAVE_API_KEY}"
      },
      "disabled": true,
      "note": "Ricerca web avanzata"
    },
    "puppeteer": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
      "disabled": true,
      "note": "Browser automation per testing"
    }
  }
}
```

### 3.2 Best Practice MCP

- Usare variabili d'ambiente per token sensibili
- Disabilitare server non necessari per progetto
- Preferire `uvx` per server Python (più veloce)

---

## 4. Workflow Agentico Integrato

### 4.1 Workflow di Sviluppo Completo

```
┌─────────────────────────────────────────────────────────────────┐
│                    WORKFLOW AGENTICO INTEGRATO                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. IDEAZIONE                                                    │
│     └─→ brainstorming skill                                      │
│         • Raffinamento requirements                              │
│         • Esplorazione alternative                               │
│         • Validazione design a chunk                             │
│                                                                   │
│  2. PIANIFICAZIONE                                               │
│     └─→ planning-with-files + writing-plans                      │
│         • task_plan.md con fasi                                  │
│         • Breakdown in task atomiche (2-5 min)                   │
│         • File paths e verification steps                        │
│                                                                   │
│  3. SVILUPPO                                                     │
│     └─→ test-driven-development                                  │
│         • RED: Scrivi test fallente                              │
│         • GREEN: Implementazione minima                          │
│         • REFACTOR: Pulizia codice                               │
│                                                                   │
│  4. DEBUG (se necessario)                                        │
│     └─→ systematic-debugging                                     │
│         • Fase 1: Riproduci                                      │
│         • Fase 2: Isola                                          │
│         • Fase 3: Identifica root cause                          │
│         • Fase 4: Fix e verify                                   │
│                                                                   │
│  5. REVIEW                                                       │
│     └─→ code-review skill                                        │
│         • Checklist automatica                                   │
│         • Verifica contro piano                                  │
│         • Report per severità                                    │
│                                                                   │
│  6. COMPLETAMENTO                                                │
│     └─→ git-workflow + verification-before-completion            │
│         • Commit strutturati                                     │
│         • Push e PR (se applicabile)                             │
│         • Cleanup task_plan.md                                   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Trigger Automatici per Skill

| Contesto | Skill Attivata | Azione |
|----------|----------------|--------|
| Nuova feature request | brainstorming | Raccogli requirements |
| Task > 3 step | planning-with-files | Crea task_plan.md |
| Prima di scrivere codice | test-driven-development | Scrivi test |
| Errore ricorrente | systematic-debugging | Avvia debug |
| Dopo implementazione | code-review | Review automatica |
| Fine task | verification-before-completion | Verifica finale |

---

## 5. Piano di Implementazione

### 5.1 Fase 1: Setup Struttura (Giorno 1)

**Task:**
1. Creare struttura directory
2. Installare agentget globalmente
3. Installare skills da superpowers

**Comandi:**
```bash
# Installa agentget
npm install -g agentget

# Installa superpowers skills
npx agentget add obra/superpowers --skills-only

# Verifica installazione
npx agentget list --skills-only
```

### 5.2 Fase 2: Skills Core (Giorni 2-3)

**Task:**
1. Configurare TDD skill con hooks
2. Configurare systematic-debugging
3. Testare workflow base

**File da creare:**
- `~/.kilocode/skills/test-driven-development/SKILL.md`
- `~/.kilocode/skills/systematic-debugging/SKILL.md`

### 5.3 Fase 3: Agenti Specializzati (Giorni 4-5)

**Task:**
1. Creare agent orchestrator
2. Creare agent code-reviewer
3. Creare agent debugger
4. Integrare con workflow

**Agenti da agentget:**
```bash
# Installa agenti specifici
npx agentget add joeyism/agentget --agent atlas        # Orchestrator
npx agentget add joeyism/agentget --agent oracle       # Architecture/Debug
npx agentget add joeyism/agentget --agent explore      # Codebase search
```

### 5.4 Fase 4: Ottimizzazione MCP (Giorno 6)

**Task:**
1. Aggiornare mcp.json con variabili ambiente
2. Configurare MCP server aggiuntivi
3. Testare integrazione

### 5.5 Fase 5: Documentazione e Testing (Giorno 7)

**Task:**
1. Documentare workflow
2. Creare esempi d'uso
3. Testare scenario end-to-end

---

## 6. Metriche di Successo

### 6.1 KPI da Tracciare

| Metrica | Target | Misurazione |
|---------|--------|-------------|
| Tempo medio per task | -30% | Confronto pre/post |
| Errori ripetuti | -50% | Tracking in task_plan.md |
| Code review coverage | 100% | Tutti i commit reviewati |
| TDD compliance | 80% | Test prima di codice |
| KV-cache hit rate | >70% | Monitoring MCP |

### 6.2 Valutazione Qualitativa

- [ ] Workflow fluido senza interruzioni
- [ ] Skills si attivano automaticamente al momento giusto
- [ ] Meno contesto perso in task lunghi
- [ ] Debugging più veloce e sistematico
- [ ] Codice più pulito e testato

---

## 7. Risorse e Riferimenti

### 7.1 Repository Analizzati

| Repo | URL | Utilità |
|------|-----|---------|
| agentget | https://github.com/joeyism/agentget | Installazione agenti/skills |
| superpowers | https://github.com/obra/superpowers | Metodologia e skills |
| skills.sh | https://skills.sh/ | Ecosistema skills |

### 7.2 Skills Leaderboard (da skills.sh)

Top skills da considerare:
1. `find-skills` - Ricerca skills
2. `brainstorming` (obra/superpowers) - 61.5K installs
3. `systematic-debugging` (obra/superpowers) - 33.6K installs
4. `test-driven-development` (obra/superpowers) - 27.9K installs
5. `writing-plans` (obra/superpowers) - 31.9K installs

### 7.3 Principi Manus (Già Implementati)

La skill `planning-with-files` esistente già implementa:
- Filesystem come memoria esterna
- Manipolazione attenzione tramite recitazione
- Tracking errori
- 3-strike error protocol

---

## 8. Prossimi Passi Immediati

### 8.1 Azioni da Eseguire Ora

```bash
# 1. Installa agentget
npm install -g agentget

# 2. Installa superpowers completo
npx agentget add obra/superpowers --all

# 3. Installa agenti specializzati
npx agentget add joeyism/agentget --agent atlas
npx agentget add joeyism/agentget --agent oracle
npx agentget add joeyism/agentget --agent explore

# 4. Verifica installazione
npx agentget list --all
```

### 8.2 Configurazione Ambiente

```bash
# Aggiungi a ~/.bashrc o ~/.zshrc
export GITHUB_TOKEN="your_token_here"
export BRAVE_API_KEY="your_key_here"  # Opzionale

# Alias utili
alias kc-skills="npx agentget list --skills-only"
alias kc-agents="npx agentget list --agents-only"
alias kc-plan="cat task_plan.md 2>/dev/null || echo 'No plan found'"
```

---

## 9. Conclusione

Questo piano fornisce un framework completo per trasformare Kilocode CLI in un ambiente di sviluppo agentico ottimizzato. L'approccio modulare permette implementazione incrementale, iniziando dalle skills ad alta priorità e espandendo gradualmente.

**Benefici attesi:**
- Sviluppo più strutturato e meno error-prone
- Workflow TDD automatico
- Debugging sistematico
- Code review integrata
- Migliore gestione del contesto

---

*Documento creato: 2026-03-18*
*Basato su analisi di: agentget, superpowers, skills.sh, configurazione Kilocode esistente*
