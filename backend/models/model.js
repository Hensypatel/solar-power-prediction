// backend/models/models.js
import mongoose from 'mongoose';
const { Schema } = mongoose;

// 1. Users Schema
const UserSchema = new Schema({
    name: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    createdAt: { type: Date, default: Date.now }
});

// 2. Locations Schema
const LocationSchema = new Schema({
    user: { type: Schema.Types.ObjectId, ref: 'User', required: true },
    latitude: { type: Number, required: true },
    longitude: { type: Number, required: true },
    city: { type: String },
    timezone: { type: String },
    createdAt: { type: Date, default: Date.now }
});

// 3. Panels Schema
const PanelSchema = new Schema({
    location: { type: Schema.Types.ObjectId, ref: 'Location', required: true },
    surfaceArea: { type: Number, required: true },
    tiltAngle: { type: Number, required: true },
    azimuthAngle: { type: Number, required: true },
    panelType: { type: String },
    createdAt: { type: Date, default: Date.now }
});

// 4. Environmental Data Schema
const EnvironmentalDataSchema = new Schema({
    location: { type: Schema.Types.ObjectId, ref: 'Location', required: true },
    timestamp: { type: Date, required: true },
    irradiance: { type: Number },
    temperature: { type: Number },
    humidity: { type: Number },
    windSpeed: { type: Number },
    cloudCover: { type: Number }
});

// 5. Predictions Schema
const PredictionSchema = new Schema({
    panel: { type: Schema.Types.ObjectId, ref: 'Panel', required: true },
    environmentalData: { type: Schema.Types.ObjectId, ref: 'EnvironmentalData', required: true },
    predictedPower: { type: Number },
    predictionTime: { type: Date },
    createdAt: { type: Date, default: Date.now }
});

// 6. Recommendations Schema
const RecommendationSchema = new Schema({
    panel: { type: Schema.Types.ObjectId, ref: 'Panel', required: true, unique: true },
    optimalTilt: { type: Number },
    optimalAzimuth: { type: Number },
    predictedPowerOptimal: { type: Number },
    createdAt: { type: Date, default: Date.now }
});

// 7. Reports Schema
const ReportSchema = new Schema({
    user: { type: Schema.Types.ObjectId, ref: 'User', required: true },
    reportType: { type: String },
    generatedAt: { type: Date, default: Date.now },
    fileUrl: { type: String }
});

// Create models
const User = mongoose.model('User', UserSchema);
const Location = mongoose.model('Location', LocationSchema);
const Panel = mongoose.model('Panel', PanelSchema);
const EnvironmentalData = mongoose.model('EnvironmentalData', EnvironmentalDataSchema);
const Prediction = mongoose.model('Prediction', PredictionSchema);
const Recommendation = mongoose.model('Recommendation', RecommendationSchema);
const Report = mongoose.model('Report', ReportSchema);

// Export models (ESM syntax)
export {
    User,
    Location,
    Panel,
    EnvironmentalData,
    Prediction, 
    Recommendation,
    Report
};
