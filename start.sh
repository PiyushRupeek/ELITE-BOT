#!/bin/zsh

BOT_DIR="/Users/piyush.kumar/botdebugger"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Rupeek Elite Bot — Starting up"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Start Ollama if not already running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "→ Starting Ollama..."
    brew services start ollama > /dev/null 2>&1
    echo "  Waiting for Ollama to be ready..."
    for i in {1..15}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
else
    echo "✓ Ollama already running"
fi

# 2. Confirm Ollama is up
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✗ Ollama failed to start. Try: brew services start ollama"
    exit 1
fi
echo "✓ Ollama is ready"

# 3. Start the bot
echo "→ Starting Elite Bot..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "$BOT_DIR"
source .venv/bin/activate
python main.py
