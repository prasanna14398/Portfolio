// script.js

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("contactForm");
  const msg = document.getElementById("formMessage");

  form.addEventListener("submit", function (e) {
    e.preventDefault();

    // Simple client-side validation (optional)
    const name = form.querySelector('input[type="text"]').value.trim();
    const email = form.querySelector('input[type="email"]').value.trim();
    const message = form.querySelector("textarea").value.trim();

    if (!name || !email || !message) {
      alert("Please fill in all fields.");
      return;
    }

    // Simulate message sent
    msg.classList.remove("hidden");
    msg.textContent = "Message sent successfully!";
    form.reset();

    // Hide message after 4 seconds
    setTimeout(() => {
      msg.classList.add("hidden");
    }, 4000);
  });
});
