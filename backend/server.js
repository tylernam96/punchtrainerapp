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
    cb(null, './uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({ storage });

// Homepage route
app.get('/', (req, res) => {
  res.json({ 
    message: 'Punch Trainer API',
    endpoints: {
      'GET /health': 'Check server status',
      'POST /analyze-video': 'Upload video for analysis'
    }
  });
});

// Health check route
app.get('/health', (req, res) => {
  res.json({ status: 'Server is running!' });
});

// Video analysis route
app.post('/analyze-video', upload.single('video'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No video file uploaded' });
  }

  const videoPath = path.resolve(req.file.path);
  console.log('=== VIDEO UPLOAD DEBUG ===');
  console.log('Original filename:', req.file.originalname);
  console.log('Saved filename:', req.file.filename);
  console.log('Video path:', videoPath);
  console.log('File size:', req.file.size, 'bytes');
  
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
    console.log('=== PYTHON SCRIPT DEBUG ===');
    console.log(`Python script exited with code: ${code}`);
    console.log('Python stdout length:', result.length);
    console.log('Python stderr length:', error.length);
    
    if (error.length > 0) {
      console.log('Python stderr (warnings):', error);
    }
    
    if (code === 0) {
      try {
        // Clean the result - remove any stderr warnings that might be mixed in
        let cleanResult = result;
        
        // Find the JSON part (starts with { and ends with })
        const jsonStart = result.indexOf('{');
        const jsonEnd = result.lastIndexOf('}') + 1;
        
        if (jsonStart >= 0 && jsonEnd > jsonStart) {
          cleanResult = result.substring(jsonStart, jsonEnd);
          console.log('Extracted clean JSON length:', cleanResult.length);
          console.log('Clean JSON starts with:', cleanResult.substring(0, 100));
        } else {
          console.log('Could not find JSON boundaries in result');
          console.log('Raw result:', result);
        }
        
        const feedback = JSON.parse(cleanResult);
        console.log('=== PARSED RESULT DEBUG ===');
        console.log('Result has keys:', Object.keys(feedback));
        console.log('Success field:', feedback.success);
        console.log('Has analysis:', !!feedback.analysis);
        
        if (feedback.analysis) {
          console.log('Analysis type:', feedback.analysis.analysis_type);
          console.log('Analysis has keys:', Object.keys(feedback.analysis));
          if (feedback.analysis.individual_punches) {
            console.log('Number of punches analyzed:', feedback.analysis.individual_punches.length);
          }
        }
        
        res.json(feedback);
      } catch (e) {
        console.log('=== JSON PARSE ERROR ===');
        console.log('Parse error:', e.message);
        console.log('Raw result length:', result.length);
        console.log('First 1000 chars of raw result:');
        console.log(result.substring(0, 1000));
        console.log('Last 500 chars of raw result:');
        console.log(result.substring(Math.max(0, result.length - 500)));
        
        // Try to find if there are multiple JSON objects
        const jsonMatches = result.match(/\{[^{}]*\}/g);
        if (jsonMatches) {
          console.log('Found potential JSON objects:', jsonMatches.length);
          jsonMatches.forEach((match, index) => {
            console.log(`JSON object ${index + 1}:`, match.substring(0, 100));
          });
        }
        
        res.status(500).json({ 
          error: 'Failed to parse analysis results', 
          details: e.message,
          raw_length: result.length,
          raw_preview: result.substring(0, 500),
          stderr_preview: error.substring(0, 200)
        });
      }
    } else {
      console.log('=== PYTHON SCRIPT FAILED ===');
      res.status(500).json({ 
        error: 'Python analysis failed', 
        details: error || 'Unknown error',
        exit_code: code 
      });
    }
  });

  python.on('error', (err) => {
    console.log('=== PYTHON SPAWN ERROR ===');
    console.log('Spawn error:', err.message);
    res.status(500).json({ 
      error: 'Failed to start Python script', 
      details: err.message 
    });
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`📁 Upload directory: ${path.resolve('./uploads')}`);
  console.log(`🐍 Python script: ${path.resolve('../analysis/analyze_punch.py')}`);
});