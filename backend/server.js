import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import connectDB from './config/db.js';
import { User, Location, Panel, EnvironmentalData, Prediction, Recommendation, Report } from './models/model.js';
import predictRoutes from './routes/predictRoutes.js';

dotenv.config();

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Connect to MongoDB
connectDB();

// Routes
app.get('/', (req, res) => {
  res.send('Solar Power Prediction API is running!');
});

// API Routes
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    message: 'Solar Power Prediction API is healthy',
    timestamp: new Date().toISOString()
  });
});

// Test database connection
app.get('/api/test-db', async (req, res) => {
  try {
    // Test if we can access the models
    const userCount = await User.countDocuments();
    res.json({ 
      status: 'success', 
      message: 'Database connection successful',
      userCount: userCount
    });
  } catch (error) {
    res.status(500).json({ 
      status: 'error', 
      message: 'Database connection failed',
      error: error.message 
    });
  }
});

// Use prediction routes
app.use('/api/predict', predictRoutes);

// Start server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
});
