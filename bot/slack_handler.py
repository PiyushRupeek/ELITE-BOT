import logging
import threading
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from bot.agent import DevAgent, HELP_TEXT
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = App(token=config.SLACK_BOT_TOKEN)
agent = DevAgent()

# Cached at first use — avoids an API call on every mention
_BOT_USER_ID: str | None = None

SLACK_MSG_LIMIT = 3800  # Slack's limit is 4000; stay safely under


def _get_bot_user_id() -> str:
    global _BOT_USER_ID
    if _BOT_USER_ID is None:
        _BOT_USER_ID = app.client.auth_test()["user_id"]
    return _BOT_USER_ID


def _clean_query(text: str, bot_user_id: str) -> str:
    return text.replace(f"<@{bot_user_id}>", "").strip()


def _post_response(say, text: str, thread_ts: str | None = None):
    """Post response, splitting into multiple messages if it exceeds Slack's limit."""
    kwargs = {"thread_ts": thread_ts} if thread_ts else {}

    if len(text) <= SLACK_MSG_LIMIT:
        say(text=text, **kwargs)
        return

    # Split on double newlines to keep markdown blocks intact
    parts = []
    current = ""
    for paragraph in text.split("\n\n"):
        if len(current) + len(paragraph) + 2 > SLACK_MSG_LIMIT:
            if current:
                parts.append(current.strip())
            current = paragraph
        else:
            current = (current + "\n\n" + paragraph) if current else paragraph
    if current:
        parts.append(current.strip())

    for i, part in enumerate(parts):
        suffix = f"\n_(part {i + 1}/{len(parts)})_" if len(parts) > 1 else ""
        say(text=part + suffix, **kwargs)


def _run_with_timeout_warning(say, query: str, thread_id: str, thread_ts: str) -> str:
    """Run agent.run() and post a 'still thinking' message if it takes > 30s."""
    result_container = {}

    def warn():
        try:
            say(text="_Still thinking... Ollama is working on it_ :hourglass_flowing_sand:", thread_ts=thread_ts)
        except Exception:
            pass

    timer = threading.Timer(30, warn)
    timer.start()
    try:
        result_container["response"] = agent.run(query, thread_id=thread_id)
    finally:
        timer.cancel()

    return result_container.get("response", ":x: No response from agent.")


@app.event("app_mention")
def handle_mention(event, say, client):
    bot_user_id = _get_bot_user_id()
    query = _clean_query(event.get("text", ""), bot_user_id)
    thread_ts = event.get("thread_ts") or event["ts"]

    if not query or query.lower() in ("help", "!help"):
        say(text=HELP_TEXT, thread_ts=thread_ts)
        return

    client.reactions_add(channel=event["channel"], timestamp=event["ts"], name="thinking_face")

    try:
        response = _run_with_timeout_warning(say, query, thread_id=thread_ts, thread_ts=thread_ts)
    except Exception as e:
        logger.exception("Agent error")
        response = f":x: Something went wrong: `{e}`"
    finally:
        try:
            client.reactions_remove(channel=event["channel"], timestamp=event["ts"], name="thinking_face")
        except Exception:
            pass  # reaction may already be gone

    _post_response(say, response, thread_ts=thread_ts)


@app.event("message")
def handle_dm(event, say):
    if event.get("channel_type") != "im" or event.get("bot_id"):
        return

    query = event.get("text", "").strip()
    if not query:
        return

    thread_ts = event.get("thread_ts") or event["ts"]

    if query.lower() in ("help", "!help"):
        say(text=HELP_TEXT)
        return

    say(text=":thinking_face: On it...")

    try:
        response = _run_with_timeout_warning(say, query, thread_id=thread_ts, thread_ts=None)
    except Exception as e:
        logger.exception("Agent error in DM")
        response = f":x: Something went wrong: `{e}`"

    _post_response(say, response)


def start():
    handler = SocketModeHandler(app, config.SLACK_APP_TOKEN)
    logger.info("Starting Slack bot in Socket Mode...")
    handler.start()
