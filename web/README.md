# TheLawSays Next.js Frontend

This directory contains the production web experience for TheLawSays. It communicates with the FastAPI backend to deliver cited legal answers and mirrors the mobile mockups from `ui-screens.png`.

## Getting Started

```bash
cd web
npm install
cp .env.local.example .env.local
npm run dev
```

Visit <http://localhost:3000>. Ensure the FastAPI service is running on the host defined in `NEXT_PUBLIC_API_BASE_URL`.

## Scripts

| Command         | Description                          |
| --------------- | ------------------------------------ |
| `npm run dev`   | Start the local development server.  |
| `npm run build` | Create a production build.           |
| `npm run start` | Serve the production build.          |
| `npm run lint`  | Run ESLint checks.                   |
| `npm run test`  | Execute component tests with Vitest. |

## Project Structure

```
web/
├── app/          # Next.js App Router routes
├── components/   # Reusable UI components (chat, sidebar, theme toggle)
├── lib/          # API clients and shared TypeScript types
└── __tests__/    # Vitest/Testing Library specs
```

## Testing

```bash
npm run test
```

The test suite uses Vitest + Testing Library with a jsdom environment.

## Styling & Theming

- Tailwind CSS with custom CSS variables powers the design system.
- `next-themes` enables a light/dark mode toggle (also available inside the sidebar).

## Production Build

After running `npm run build`, deploy the generated `.next` output using your preferred platform. Ensure environment variables (`NEXT_PUBLIC_API_BASE_URL`) are provided at deploy time.
