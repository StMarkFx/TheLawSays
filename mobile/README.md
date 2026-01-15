# TheLawSays Mobile (React Native)

This Expo React Native app mirrors the core chat experience from the Next.js frontend.

## Features
- Pinned header with sidebar + new chat actions
- Chat composer with inline send
- Assistant markdown rendering
- Citation accordion for retrieved chunks
- Feedback + copy actions

## Setup

```bash
cd mobile
npm install
cp .env.example .env
```

Update `EXPO_PUBLIC_API_BASE_URL` in `.env` to point at your API.

- Android emulator: `http://10.0.2.2:8000`
- iOS simulator: `http://localhost:8000`
- Physical device: use your machine IP (e.g. `http://192.168.1.20:8000`)

## Run

```bash
npm run start
```

## Notes
- The UI is dark-first to match the web app.
- Clipboard, fonts, and gradients use Expo modules.
