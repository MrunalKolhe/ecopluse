# 🌱 EcoPulse – Smart Environmental Intelligence Platform

EcoPulse is a Green-Tech full-stack web platform designed to promote environmental awareness and sustainable living by combining real-time environmental data, user behavior tracking, and AI-driven insights.

---

## 🚀 Why EcoPulse?

Air pollution, climate change, and unsustainable daily habits are growing concerns globally.

Most existing platforms:
- Only display environmental data  
- Do NOT connect daily human behavior with environmental impact  

👉 EcoPulse bridges this gap by:
- Showing live environmental data  
- Tracking daily user activities  
- Converting them into meaningful insights  

---

## 💡 Solution Overview

EcoPulse acts as a **digital pulse of the environment**, combining:

- 🌦 Weather data  
- 🌫 Air Quality Index (AQI)  
- 🧠 User daily behavior  
- 📊 Analytics & trends  
- 🗺 Geographical heatmaps  

---

## ✨ Key Features

### 🔐 Authentication
- City-based login system  
- Personalized user tracking  

---

### 📊 Smart Dashboard
- Real-time weather (Temperature, Humidity, Wind, UV)  
- Live AQI with health categories  
- Green Index Score (eco rating system)  

---

### 📝 Daily Input System (Core Feature 🚀)
Users log:
- AC/Fan usage  
- Water consumption  
- Outdoor exposure  
- Transport mode  
- Waste segregation  

➡️ Inputs are **locked after submission**  
➡️ Used for impact analysis  

---

### 📈 Analytics & Insights
- Personalized eco-score  
- Habit-based environmental impact  
- Progress tracking over time  

---

### 🧠 AI + Trends
- AQI forecasting (ML-based)  
- Temperature & pollution trends  
- Data-driven recommendations  

---

### 🗺 India Heatmap
- Visual pollution intensity across regions  
- Easy understanding of environmental distribution  

---

### 🚯 Civic Issue Reporting System (ADVANCED FEATURE 🔥)
- Users can report real-world issues (garbage, potholes, etc.)
- Upload image + description + location  
- AI-assisted classification  

#### 🏛 Workflow:
- User → Admin → Government Department  
- Status tracking:
  - Pending  
  - In Progress  
  - Resolved  

---

### 🏆 Gamification & Leaderboard
- Users earn points for eco-friendly habits  
- Ranking system for:
  - Sustainable users  
  - Civic contributors  

---

### ⚠ Environmental Awareness Module
- PM2.5, PM10, CO, NO₂, SO₂, O₃ explained  
- Health impact insights  

---

## 📌 Project Impact

EcoPulse encourages sustainable habits by connecting
daily user activities with measurable environmental impact.

The platform helps users:
- Understand pollution levels
- Improve eco-friendly behavior
- Report civic issues
- Track long-term sustainability goals


## 🛠 Tech Stack

### 💻 Frontend
- Next.js (App Router)  
- TypeScript  
- Tailwind CSS  
- ShadCN UI  

### ⚙️ Backend
- Python (FastAPI)  

### 🧠 Machine Learning & Analytics
- Python
- ONNX Runtime
- Data Processing
- AQI Forecasting Models

### 🌐 APIs
- OpenWeather API  
- Air Pollution API  

---

## 🏗️ System Architecture

```text
                    ┌──────────────────┐
                    │      Users       │
                    └────────┬─────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │   Next.js Frontend     │
                │  Dashboard & UI Layer  │
                └────────┬───────────────┘
                         │ API Requests
                         ▼
                ┌────────────────────────┐
                │    FastAPI Backend     │
                │ Business Logic Layer   │
                └────────┬───────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼

 ┌────────────┐   ┌────────────┐   ┌─────────────┐
 │ OpenWeather│   │ AQI APIs   │   │ ML Models   │
 │    API     │   │ Pollution  │   │ Forecasting │
 └────────────┘   └────────────┘   └─────────────┘

                         │
                         ▼

                ┌────────────────────────┐
                │ User Analytics Engine  │
                │ Eco Score Calculation  │
                └────────┬───────────────┘
                         │
                         ▼

                ┌────────────────────────┐
                │ Leaderboards & Reports │
                └────────────────────────┘

```
## 📂 Project Structure

frontend/
backend/
ml-model/
public/
uploads/

README.md

---

## 📸 Screenshots

### 📊 Dashboard
<img src="./screenshots-dashboard.png" width="800"/>

### 📝 Daily Input System
<img src="./screenshots-daily-input.png" width="800"/>

### 🚯 Civic Issue Admin Panel (Core Feature 🚀)
<img src="./screenshots-civic-admin.png" width="800"/>
---

## 🚀 Challenges Solved

- Designed a complete User → Admin → Government issue escalation workflow.
- Integrated weather and AQI APIs into a unified dashboard.
- Developed a daily habit tracking mechanism with locked submissions.
- Built an environmental impact scoring system based on user activities.
- Visualized pollution trends using interactive heatmaps and analytics.
- Managed real-time environmental data efficiently within a full-stack architecture.

## 🎯 Use Cases

- Environmental awareness  
- Smart travel planning  
- Urban pollution monitoring  
- Sustainability education  
- Government & NGO insights  

---

## 🚧 Future Enhancements

- AI-based pollution prediction  
- Mobile application  
- Government dashboard integration  
- Real-time alerts for AQI  

---

## 👨‍💻 Author

Mrunal Kolhe  
GitHub: https://github.com/tylrx404  

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
