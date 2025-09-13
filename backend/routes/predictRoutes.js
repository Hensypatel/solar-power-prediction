// backend/routes/predictRoutes.js
import express from 'express';
import { getPrediction, getHistory, exportData } from '../controllers/predictController.js';

const router = express.Router();

router.post('/predict', getPrediction);
router.get('/history', getHistory);
router.get('/export', exportData);

export default router;
