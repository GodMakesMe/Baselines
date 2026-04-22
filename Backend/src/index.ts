import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import scriptRunRouter from './Routes/script-run';
import inkDetectionRouter from './Routes/ink-detection';

const app = express();
const port = 2632;

const defaultAllowedOrigins = [
  'http://localhost:3000',
  'http://127.0.0.1:3000',
  'https://godmakesme.com',
  'https://www.godmakesme.com',
  'https://cv.godmakesme.com',
  'https://api.godmakesme.com',
];

const envAllowedOrigins = (process.env.CORS_ORIGINS ?? '')
  .split(',')
  .map((origin) => origin.trim())
  .filter(Boolean);

const allowedOrigins = new Set([...defaultAllowedOrigins, ...envAllowedOrigins]);

app.use(cors({
  origin: (origin, callback) => {
    if (!origin || allowedOrigins.has(origin)) {
      callback(null, true);
      return;
    }

    callback(new Error(`Origin not allowed by CORS: ${origin}`));
  },
  credentials: true,
}));
app.use(express.json());
app.use('/api/scripts', scriptRunRouter);
app.use('/api/ink', inkDetectionRouter);





app.get('/', (_req, res) => {
  res.json({
    message: 'Hello from backend',
    service: 'api.godmakesme.com',
  });
});

app.get('/health', (_req, res) => {
  res.json({ status: 'ok' });
});

app.listen(port, '0.0.0.0', () => {
  console.log(`Server is running on port ${port}`);
});
