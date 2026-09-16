from pathlib import Path
from typing import Any, Literal

from ollama import Client


class OllamaAssistant:
    """Client wrapper for local Ollama text and image interactions.

    ``chat`` maintains a conversation history on this instance. ``infer``
    sends an isolated request and leaves that history unchanged. Both methods
    can optionally include local image paths and request JSON output.
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        host: str = "http://localhost:11434",
        system_prompt: str | None = None,
    ) -> None:
        """Create an assistant connected to an Ollama server.

        Args:
            model: Name of the locally available Ollama model.
            host: Base URL of the Ollama server.
            system_prompt: Optional instruction included in every request.
        """

        self._model = model
        self._client = Client(host=host)
        self._messages: list[dict[str, Any]] = []

        if system_prompt:
            self._messages.append({
                "role": "system",
                "content": system_prompt,
            })

    def _build_message(
        self,
        prompt: str,
        images: list[str | Path] | None = None,
        files: list[str | Path] | None = None,
    ) -> dict[str, Any]:
        """Build an Ollama user message from text, images, and text files."""

        message: dict[str, Any] = {
            "role": "user",
            "content": prompt,
        }
        if images:
            message["images"] = [str(image) for image in images]
        if files:
            message["content"] = self._append_files(prompt, files)
        return message

    def _append_files(
        self,
        prompt: str,
        files: list[str | Path],
    ) -> str:
        """Append UTF-8 text files to a prompt with explicit file markers."""

        sections = [prompt]
        for file in files:
            path = Path(file)
            content = path.read_text(encoding="utf-8")
            sections.append(
                f"\n--- BEGIN FILE: {path.name} ---\n"
                f"{content}\n"
                f"--- END FILE: {path.name} ---"
            )
        return "\n".join(sections)

    def _request(
        self,
        messages: list[dict[str, Any]],
        response_format: Literal["json"] | dict[str, Any] | None = None,
    ) -> str:
        """Send messages to Ollama and return the assistant text.

        This method only performs the API request. It does not modify the
        stored conversation history; the public methods decide whether a
        request belongs to the ongoing conversation.
        """

        response = self._client.chat(
            model=self._model,
            messages=messages,
            format=response_format,
        )

        content = response.message.content
        if content is None:
            raise RuntimeError("Ollama returned no response content.")
        return content

    def chat(
        self,
        prompt: str,
        images: list[str | Path] | None = None,
        files: list[str | Path] | None = None,
        response_format: Literal["json"] | dict[str, Any] | None = None,
    ) -> str:
        """Send a message and preserve the conversation history.

        The new user message and Ollama's response are stored in this
        instance. Later calls to ``chat`` can therefore refer to earlier
        messages. ``images`` accepts paths as strings or ``Path`` objects.

        Args:
            prompt: User question or instruction.
            images: Optional local images to attach to the request.
            files: Optional UTF-8 text files such as code or JSON.
            response_format: ``"json"`` or a JSON schema for structured
                output. If omitted, Ollama returns normal text.

        Returns:
            The assistant response as text.
        """

        message = self._build_message(prompt, images, files)
        self._messages.append(message)
        content = self._request(self._messages, response_format)
        self._messages.append({"role": "assistant", "content": content})
        return content

    def infer(
        self,
        prompt: str,
        images: list[str | Path] | None = None,
        files: list[str | Path] | None = None,
        response_format: Literal["json"] | dict[str, Any] | None = None,
    ) -> str:
        """Run one isolated request without changing the chat history.

        Only the configured system prompt and this request are sent to the
        model. Previous chat messages are deliberately excluded, making this
        suitable for independent receipt or image analysis.

        Args:
            prompt: User question or instruction for this single request.
            images: Optional local images to attach to the request.
            files: Optional UTF-8 text files such as code or JSON.
            response_format: ``"json"`` or a JSON schema for structured
                output. If omitted, Ollama returns normal text.

        Returns:
            The assistant response as text.
        """

        message = self._build_message(prompt, images, files)
        system_messages = [
            message
            for message in self._messages
            if message.get("role") == "system"
        ]
        return self._request(system_messages + [message], response_format)

    def reset(self) -> None:
        """Clear chat messages while keeping the system prompt and model."""

        self._messages = [
            message
            for message in self._messages
            if message.get("role") == "system"
        ]

    def interactive_chat(self) -> None:
        """Start a terminal chat using the persistent ``chat`` history.

        Enter ``/reset`` to remove the conversation messages, or ``/exit``
        to leave the interactive session. EOF and Ctrl+C also end the chat.
        """

        print(f"Chat with {self._model}. Type '/exit' to leave or '/reset' to clear.")
        while True:
            try:
                prompt = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if prompt == "/exit":
                return
            if prompt == "/reset":
                self.reset()
                print("Conversation reset.")
                continue
            if not prompt:
                continue

            print(f"assistant> {self.chat(prompt)}")
