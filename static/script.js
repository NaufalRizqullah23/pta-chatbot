document.addEventListener("DOMContentLoaded", function () {
  const form = document.querySelector("form");

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    const messageInput = document.querySelector('input[name="message"]');
    const message = messageInput.value;
    messageInput.value = "";

    fetch("/chat", {
      method: "POST",
      body: new URLSearchParams({ message }),
    })
      .then(function (response) {
        return response.text();
      })
      .then(function (response) {
        const chatContainer = document.querySelector("#chat-container");
        const chatBubble = document.createElement("div");
        chatBubble.classList.add("chat-bubble");
        chatBubble.textContent = response;
        chatContainer.appendChild(chatBubble);
      })
      .catch(function (error) {
        console.error("Error:", error);
      });
  });
});
