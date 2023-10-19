const chatbotToggler = document.querySelector(".chatbot-toggler");
const closeBtn = document.querySelector(".close-btn");
const chatbox = document.querySelector(".chatbox");
const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");
const apiUrl = 'http://localhost:5000/get_bot_response';

emailjs.init("oZPkPuKFgy0rUdFkv");

let userMessage = null; // Variable to store user's message
const inputInitHeight = chatInput.scrollHeight;
const chatHistory = [];

const createChatLi = (message, className, isUser = false) => {
    // Create a chat <li> element with passed message and className
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", `${className}`);
    let chatContent = className === "outgoing" ? `<p></p>` : `<span><img src="sbclogo.jpg" style="width: 35px; height: 35px;" alt="sbc"></span><p></p>`;
    
    if (isUser) {
        chatContent = `<p></p><span><img src="clipart.png" style="width: 35px; height: 30px; padding-left: 10px; margin-top: 12px" alt="user"></span>`;
    }
    
    chatLi.innerHTML = chatContent;
    chatLi.querySelector("p").textContent = message;
    return chatLi; // return chat <li> element
}

const generateResponse = (chatElement) => {
    const API_URL = "https://api.openai.com/v1/chat/completions";
    const messageElement = chatElement.querySelector("p");

    // Define the properties and message for the API request
    const requestOptions = {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${API_KEY}`
        },
        body: JSON.stringify({
            model: "gpt-3.5-turbo",
            messages: [{role: "user", content: userMessage}],
        })
    }

    // Send POST request to API, get response and set the reponse as paragraph text
    fetch(API_URL, requestOptions).then(res => res.json()).then(data => {
        messageElement.textContent = data.choices[0].message.content.trim();
    }).catch(() => {
        messageElement.classList.add("error");
        messageElement.textContent = "Oops! Something went wrong. Please try again.";
    }).finally(() => chatbox.scrollTo(0, chatbox.scrollHeight));
}

const handleChat = () => {
    userMessage = chatInput.value.trim(); // Get user entered message and remove extra whitespace
    if(!userMessage) return;

    chatHistory.push({ role: "user", content: userMessage });

    fetch(apiUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_message: userMessage }),
    })
        .then((response) => response.json())
        .then((data) => {
            const botResponse = data.bot_response;
            chatHistory.push({ role: "assistant", content: botResponse });
            // Display botResponse in the chat interface
        })
        .catch((error) => {
            console.error('API request error:', error);
        });
}

chatInput.addEventListener("input", () => {
    // Adjust the height of the input textarea based on its content
    chatInput.style.height = `${inputInitHeight}px`;
    chatInput.style.height = `${chatInput.scrollHeight}px`;
});

chatInput.addEventListener("keydown", async (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const userMessage = chatInput.value.trim();
  
      if (userMessage) {
        // Display the user message in the chat
        const userChatLi = createChatLi(userMessage, "outgoing", true);
        chatbox.appendChild(userChatLi);
        chatInput.value = "";
  
        const thinkingChatLi = createChatLi('Thinking...', 'incoming');
        chatbox.appendChild(thinkingChatLi);
        chatbox.scrollTop = chatbox.scrollHeight;

        // Send the user message to the server
        try {
          const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ user_message: userMessage }),
          });
  
          if (response.ok) {
            const data = await response.json();
            const botResponse = data.bot_response;

            chatbox.removeChild(thinkingChatLi);
            chatbox.scrollTop = chatbox.scrollHeight;
  
            // Display the bot's response in the chat
            const botChatLi = createChatLi('', "incoming");
            chatbox.appendChild(botChatLi);

            const words = botResponse.split(' ');

            const printWords = async () => {
                for (const word of words) {
                    botChatLi.querySelector("p").textContent += word + ' ';
                    await new Promise(resolve => setTimeout(resolve, 140));
                }
                chatbox.scrollTop = chatbox.scrollHeight;
            };

            printWords().then(() => {
                //
            });
          } else {
            throw new Error('Server response not ok');
          }
        } catch (error) {
          console.error('Error sending/receiving data:', error);
          // Handle the error, e.g., display an error message in the chat
          const errorChatLi = createChatLi('Oops! Something went wrong.', 'incoming');
          chatbox.appendChild(errorChatLi);
        }
      }
    }
});

document.getElementById("sendToEmail").addEventListener("click", function () {
    const userEmail = prompt("Please enter your email address:");

    if (userEmail) {
        // Get the last two messages from chat history
        const lastTwoChats = chatHistory.slice(-2);

        lastTwoChats.forEach((chat) => {
            emailContent += `${chat.role === "user" ? "User Prompt:" : "Bot Response:"} ${chat.content}\n`;
        });

        // Send the email using Email.js
        emailjs.send("service-id", "template-id", {
            to: userEmail,
            message: emailContent,
        }).then(function (response) {
            console.log("Email sent:", response);
            alert("Email sent successfully!");
            console.log(emailContent);
        }, function (error) {
            console.error("Email sending error:", error);
            alert("Email sending failed. Please try again later.");
        });
    }
});

sendChatBtn.addEventListener("click", handleChat);
closeBtn.addEventListener("click", () => document.body.classList.remove("show-chatbot"));
chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));