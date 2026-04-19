#!/bin/zsh

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Rupeek Elite Bot — Shutting down"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Stop the bot (kill python main.py process)
BOT_PID=$(pgrep -f "python main.py")
if [ -n "$BOT_PID" ]; then
    kill "$BOT_PID"
    echo "✓ Elite Bot stopped (pid $BOT_PID)"
else
    echo "  Elite Bot was not running"
fi

# 2. Stop Ollama service
brew services stop ollama > /dev/null 2>&1
echo "✓ Ollama stopped"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  All services stopped"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
