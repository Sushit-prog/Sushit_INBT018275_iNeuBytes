/**
 * Front-end JavaScript for the AI Mentor Chatbot.
 * Vanilla JS — no frameworks.
 */

(function () {
    "use strict";

    // ---- DOM references --------------------------------------------------
    const chatContainer = document.getElementById("chat-container");
    const inputField    = document.getElementById("message-input");
    const sendBtn       = document.getElementById("send-btn");

    // ---- Helpers ----------------------------------------------------------

    /** Append a message bubble to the chat area. */
    function appendMessage(role, text, sources) {
        const div = document.createElement("div");
        div.className = "message " + role;

        const textPara = document.createElement("p");
        textPara.textContent = text;
        div.appendChild(textPara);

        // Show source filenames below bot responses
        if (role === "bot" && sources && sources.length > 0) {
            const sourceEl = document.createElement("p");
            sourceEl.className = "sources";
            sourceEl.textContent = "Sources: " + sources.join(", ");
            div.appendChild(sourceEl);
        }

        chatContainer.appendChild(div);
        scrollToBottom();
    }

    /** Show the animated loading-indicator bubble. */
    function showLoading() {
        const container = document.createElement("div");
        container.id    = "loading-indicator";
        container.className = "loading-dots";

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement("span");
            container.appendChild(dot);
        }

        chatContainer.appendChild(container);
        scrollToBottom();
    }

    /** Remove the loading-indicator bubble. */
    function hideLoading() {
        const indicator = document.getElementById("loading-indicator");
        if (indicator) indicator.remove();
    }

    /** Scroll the chat container to the bottom. */
    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // ---- Core logic -------------------------------------------------------

    async function sendMessage() {
        const raw = inputField.value;
        const text = raw.trim();

        if (!text) return;

        // Show the user's message immediately
        appendMessage("user", text);
        inputField.value = "";
        setInputEnabled(false);

        // Show loading dots
        showLoading();

        try {
            const response = await fetch("/respond", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text }),
            });

            hideLoading();

            const data = await response.json();

            if (response.ok && data.response) {
                appendMessage("bot", data.response, data.sources || []);
            } else {
                appendMessage("error", data.error || "Something went wrong. Please try again.");
            }
        } catch (_err) {
            hideLoading();
            appendMessage("error", "Network error. Please check your connection and try again.");
        } finally {
            setInputEnabled(true);
            inputField.focus();
        }
    }

    /** Enable or disable the input controls while a request is in flight. */
    function setInputEnabled(enabled) {
        inputField.disabled = !enabled;
        sendBtn.disabled    = !enabled;
        if (enabled) {
            inputField.focus();
        }
    }

    // ---- Event binding ----------------------------------------------------

    sendBtn.addEventListener("click", sendMessage);

    inputField.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Focus the input on page load
    inputField.focus();
})();
