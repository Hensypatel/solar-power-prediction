// backend/controllers/predictController.js
export const getPrediction = async (req, res) => {
    try {
      const { location, panelDetails } = req.body;
      // Call your formula or ML service here
      const predictedPower = 123; // example
      res.json({ predictedPower });
    } catch (err) {
      res.status(500).json({ message: err.message });
    }
  };
  
  export const getHistory = async (req, res) => {
    res.json({ data: [] }); // placeholder
  };
  
  export const exportData = async (req, res) => {
    res.json({ message: 'Export CSV/PDF' }); // placeholder
  };
  