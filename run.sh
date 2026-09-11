#!/usr/bin/env bash
set -euo pipefail

# Ravi voice agent — dev runner.
#   ./run.sh            start the server on :8000
#   ./run.sh test       run the vendor self-test
#   ./run.sh tail       follow the log
#   ./run.sh call +91…  place an outbound call

PORT="${PORT:-8000}"

case "${1:-serve}" in
  test)
    python -m tools.selftest
    ;;
  tail)
    tail -f logs/agent.log
    ;;
  call)
    curl -sS -X POST "http://localhost:${PORT}/call/outbound" \
      -H 'Content-Type: application/json' \
      -d "{\"to_number\":\"${2:?usage: ./run.sh call +91XXXXXXXXXX}\"}" | python -m json.tool
    ;;
  serve|*)
    # One worker only. Each call holds websockets to Teler, Deepgram and Sarvam
    # plus an asyncio playout clock; multiple workers just fragment that state.
    exec uvicorn app.server:app --host 0.0.0.0 --port "${PORT}" --workers 1 --log-level warning
    ;;
esac
