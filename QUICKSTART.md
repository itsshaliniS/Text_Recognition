# 🚀 Quick Start Guide

Get your OCR system running in 5 minutes!

## 📋 Step 1: Install Dependencies

```bash
cd OCR-Project
pip install -r requirements.txt
```

This installs all required packages including PyTorch, Flask, MLflow, etc.

---

## 🎯 Step 2: Choose Your Path

### Option A: Run Demo (Without Training)

Perfect for testing the system immediately!

```bash
python app.py
```

- Opens web interface at `http://localhost:5000`
- Uses untrained model (for demo purposes)
- Upload any handwritten image to test

### Option B: Train Model (With Real Data)

For production-quality results:

1. **Download IAM Dataset**
   - Visit: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
   - Register and download `words.tgz`
   - Extract to `OCR-Project/data/`

2. **Run Training**
   ```bash
   cd src
   python train.py
   ```

3. **Monitor Progress**
   - Watch console for training metrics
   - Run `mlflow ui` in another terminal to view experiments

4. **Deploy Trained Model**
   ```bash
   cd ..
   python app.py
   ```

---

## 🌐 Step 3: Use the Web Interface

1. Open browser: `http://localhost:5000`
2. Upload an image with handwritten text
3. Click "Extract Text"
4. View and copy results!

---

## 🔥 Quick Commands Reference

```bash
# Install everything
pip install -r requirements.txt

# Train model
cd src && python train.py

# Test model architecture
cd src && python model.py

# Run web app
python app.py

# View MLflow experiments
mlflow ui

# Run with Gunicorn (production)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## 🎓 Training Configuration

Edit `src/train.py` to customize:

```python
config = {
    'num_epochs': 20,        # More epochs = better accuracy
    'batch_size': 32,        # Increase if you have GPU memory
    'learning_rate': 0.001,  # Adjust based on loss curve
}
```

---

## 📊 Expected Results

### Training Metrics (after 20 epochs on IAM):
- Train Loss: ~0.5
- Validation Loss: ~0.8
- Character Error Rate (CER): ~10-15%
- Word Error Rate (WER): ~25-30%

### Inference:
- Processing Time: ~50ms per image (GPU)
- Supported Characters: Letters, digits, punctuation
- Image Size: Automatically resized to 32x128

---

## 🐛 Troubleshooting

### Issue: "CUDA not available"
**Solution**: PyTorch will use CPU automatically. Training will be slower but functional.

### Issue: "Module not found"
**Solution**: Make sure you're in the correct directory and virtual environment is activated.
```bash
# Activate venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Reinstall
pip install -r requirements.txt
```

### Issue: "Port 5000 already in use"
**Solution**: Change port in `app.py`:
```python
app.run(host='0.0.0.0', port=8080)  # Use different port
```

### Issue: "Dataset not found"
**Solution**: The app creates dummy data automatically for testing. Download IAM dataset for real training.

---

## 💡 Tips

1. **GPU Training**: Training on GPU is 10-50x faster. Use Google Colab if you don't have GPU.
2. **Start Small**: Test with few epochs first (`num_epochs: 2`) to verify everything works.
3. **Monitor Memory**: Reduce `batch_size` if you run out of GPU memory.
4. **Save Often**: Model checkpoints are saved automatically every epoch.

---

## 🎯 Next Steps

After getting started:

1. ✅ **Explore MLflow UI**: View training metrics and experiments
2. ✅ **Try DAGsHub**: Connect for remote experiment tracking
3. ✅ **Customize Model**: Adjust architecture in `src/model.py`
4. ✅ **Deploy Online**: Use Heroku, AWS, or Google Cloud
5. ✅ **Add Features**: Batch processing, PDF support, etc.

---

## 📚 Full Documentation

For detailed information, see:
- **README.md**: Complete documentation
- **mlflow.yaml**: Configuration options
- **Code Comments**: Inline documentation in all files

---

## 🆘 Need Help?

- Check README.md for detailed instructions
- Review code comments for implementation details
- Open an issue on GitHub
- Check PyTorch/MLflow documentation

---

**Happy Coding! 🚀**

---

## ⚡ Ultra-Quick Start (One Command)

```bash
pip install -r requirements.txt && python app.py
```

That's it! Your OCR system is now running on `http://localhost:5000` 🎉

