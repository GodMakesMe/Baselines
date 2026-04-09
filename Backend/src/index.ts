import express from 'express';
import cors from 'cors';
import scriptRunRouter from './Routes/script-run';

const app = express();
const port = 2632;

app.use(cors({
  origin: 'http://localhost:3000',
  credentials: true,
}));
app.use(express.json());
app.use('/api/scripts', scriptRunRouter);





app.get('/', (req, res) => {
  res.json({ message: 'Welcome to the TypeScript Backend' });
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
