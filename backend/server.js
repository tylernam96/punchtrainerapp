const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const { spawn } = require('child_process');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());

// File upload setup
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, './uploads/');  // Changed from '../videos/' to './uploads/'
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({ storage });

// Routes
app.post('/analyze-video', upload.single('video'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No video file uploaded' });
  }

  const videoPath = path.resolve(req.file.path);  // This should now point to backend/uploads/
  console.log('Video path:', videoPath);  // Add this for debugging
  
  // Call Python analysis script
  const python = spawn('python3', [
    path.resolve('../analysis/analyze_punch.py'), 
    videoPath
  ]);

  let result = '';
  let error = '';

  python.stdout.on('data', (data) => {
    result += data.toString();
  });

  python.stderr.on('data', (data) => {
    error += data.toString();
  });

  python.on('close', (code) => {
    if (code === 0) {
      try {
        const feedback = JSON.parse(result);
        res.json({ success: true, feedback });
      } catch (e) {
        res.status(500).json({ error: 'Failed to parse analysis results' });
      }
    } else {
      res.status(500).json({ error: error || 'Analysis failed' });
    }
  });
});

app.get('/health', (req, res) => {
  res.json({ status: 'Server is running!' });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});