const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const chatContainer = document.getElementById("chat-container");

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const message = messageInput.value;

  //display user message
  displayMessage("user", message);

  //clear the input field
  messageInput.value = "";

  //send the message to the server and recieve the bot response
  fetch("/chat", {
    method: "POST",
    body: new URLSearchParams({
      message: message,
    }),
  })
    .then((response) => response.text())
    .then((response) => {
      //display bot response
      displayMessage("bot", response);
    });
});

function ScrollToBottom() {
  window.scrollTo(0, document.body.scrollHeight);
}

function displayMessage(sender, message) {
  const messageElement = document.createElement("div");
  messageElement.classList.add(
    "flex",
    "mb-2",
    sender === "user" ? "justify-end" : "justify-start"
  );

  const bubbleElement = document.createElement("div");
  bubbleElement.classList.add(
    "py-2",
    "px-4",
    "rounded-lg",
    "max-w-lg",
    "break-words",
    sender === "user" ? "user-bubble" : "bot-bubble"
  );

  bubbleElement.textContent = message;

  messageElement.appendChild(bubbleElement);
  chatContainer.appendChild(messageElement);
  ScrollToBottom();
}
