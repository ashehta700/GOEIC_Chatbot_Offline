# 🔄 MIGRATION CHECKLIST
## From OpenAI/Google APIs to Fully Offline System

Use this checklist to migrate your GOEIC RAG system step by step.

---

## 📅 PHASE 1: PREPARATION (Day 1)

### ☑️ Server Preparation

- [ ] **Verify server specs**
  ```bash
  # Check CPU
  lscpu | grep "Model name"
  
  # Check RAM
  free -h
  
  # Check GPU
  nvidia-smi
  
  # Check disk space
  df -h
  ```
  Expected: 6+ cores, 30GB RAM, RTX 4070 Ti SUPER, 100GB+ free

- [ ] **Backup current system**
  ```bash
  # Backup database
  docker exec weaviate sh -c 'tar -czf - /var/lib/weaviate' > weaviate_backup.tar.gz
  
  # Backup code
  tar -czf old_system_backup.tar.gz /path/to/old/system
  
  # Backup .env
  cp .env .env.backup
  ```

- [ ] **Update system packages**
  ```bash
  sudo apt update && sudo apt upgrade -y
  ```

### ☑️ Download Files

- [ ] **Get all offline system files**
  - main_offline.py
  - smart_scraper_offline.py
  - smart_excel_uploader_offline.py
  - requirements_offline.txt
  - .env.example
  - deploy.sh
  - docker-compose.yml

- [ ] **Copy public folder from old system**
  ```bash
  cp -r /path/to/old/system/public ./public
  ```

---

## 📅 PHASE 2: INSTALLATION (Day 1-2)

### ☑️ Install Ollama

- [ ] **Download and install**
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

- [ ] **Start Ollama service**
  ```bash
  ollama serve &
  ```

- [ ] **Pull Qwen 2.5 14B model** (⏰ This takes 20-40 minutes)
  ```bash
  ollama pull qwen2.5:14b
  ```

- [ ] **Verify installation**
  ```bash
  ollama list
  # Should show: qwen2.5:14b
  ```

### ☑️ Setup Weaviate

- [ ] **Stop old Weaviate (if running)**
  ```bash
  docker-compose down
  ```

- [ ] **Create new docker-compose.yml**
  - Use the provided file

- [ ] **Start Weaviate**
  ```bash
  docker-compose up -d
  ```

- [ ] **Wait for startup** (30 seconds)
  ```bash
  curl http://localhost:8080/v1/meta
  # Should return JSON
  ```

### ☑️ Python Environment

- [ ] **Create virtual environment**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- [ ] **Upgrade pip**
  ```bash
  pip install --upgrade pip setuptools wheel
  ```

- [ ] **Install dependencies**
  ```bash
  pip install -r requirements_offline.txt
  ```

- [ ] **Install PyTorch with CUDA**
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

- [ ] **Verify GPU support**
  ```bash
  python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
  # Should print: CUDA: True
  ```

### ☑️ Configuration

- [ ] **Create .env file**
  ```bash
  cp .env.example .env
  nano .env
  ```

- [ ] **Set correct values**
  ```bash
  OLLAMA_HOST=http://localhost:11434
  OLLAMA_MODEL=qwen2.5:14b
  OLLAMA_TIMEOUT=180
  EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
  WHISPER_MODEL_SIZE=base
  WEAVIATE_HOST=localhost
  ```

- [ ] **Create directories**
  ```bash
  mkdir -p logs uploads public
  ```

---

## 📅 PHASE 3: DATA MIGRATION (Day 2-3)

### ☑️ Option A: Fresh Scraping (Recommended)

- [ ] **Run smart scraper**
  ```bash
  python3 smart_scraper_offline.py
  ```
  ⏰ Estimated time: 2-4 hours for 6000 pages

- [ ] **Check results**
  ```bash
  # Check summary file
  cat scrape_summary_*.json
  
  # Verify database
  curl http://localhost:8000/api/admin/database-stats
  ```

### ☑️ Option B: Migrate Existing Data

- [ ] **Export from old Weaviate**
  ```bash
  # This needs to be done from old system
  # Use weaviate-client to export all objects
  ```

- [ ] **Re-index with local embeddings**
  ```bash
  # Run provided migration script
  python3 reindex_with_local_embeddings.py
  ```

### ☑️ Option C: Excel + PDFs

- [ ] **Prepare Excel file**
  - path: sheet_new.xlsx
  - Columns: url, title, category, language, path

- [ ] **Prepare Word docs**
  - Place in accessible directory
  - Update BASE_DOCS_DIR in .env

- [ ] **Run uploader**
  ```bash
  export EXCEL_FILE_PATH="sheet_new.xlsx"
  export BASE_DOCS_DIR="/path/to/docs"
  python3 smart_excel_uploader_offline.py
  ```

- [ ] **Verify upload**
  ```bash
  cat upload_summary_*.json
  ```

---

## 📅 PHASE 4: TESTING (Day 3)

### ☑️ Basic Tests

- [ ] **Start application**
  ```bash
  python3 main_offline.py
  ```

- [ ] **Check health endpoint**
  ```bash
  curl http://localhost:8000/health
  ```
  Expected: `"status": "healthy"`

- [ ] **Test chat endpoint** (Arabic)
  ```bash
  curl -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"question": "ما هي الهيئة؟", "language": "ar"}'
  ```

- [ ] **Test chat endpoint** (English)
  ```bash
  curl -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"question": "What is GOEIC?", "language": "en"}'
  ```

- [ ] **Test chat endpoint** (French)
  ```bash
  curl -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"question": "Qu'\''est-ce que GOEIC?", "language": "fr"}'
  ```

### ☑️ Quality Tests

- [ ] **Test jobs query**
  ```bash
  curl -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"question": "ما هي الوظائف المتاحة؟", "language": "ar"}'
  ```
  Expected: Live job data from website

- [ ] **Test training query**
  ```bash
  curl -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"question": "ما هي الدورات التدريبية؟", "language": "ar"}'
  ```
  Expected: Training course details

- [ ] **Test service query**
  ```bash
  curl -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"question": "كيف أحجز موعد؟", "language": "ar"}'
  ```
  Expected: Booking instructions

- [ ] **Compare with old system responses**
  - Take 10 sample questions
  - Query both old and new systems
  - Compare quality (should be 90-95% similar)

### ☑️ Admin Panel Tests

- [ ] **Login to admin panel**
  - http://localhost:8000/admin
  - Username: admin
  - Password: goeic2026

- [ ] **View analytics**
  - Check device stats
  - Check language stats
  - Check country stats

- [ ] **Test database stats**
  ```bash
  curl -X GET http://localhost:8000/api/admin/database-stats \
    -H "Cookie: session_token=YOUR_SESSION_TOKEN"
  ```

- [ ] **Test background scraping**
  - Click "Scrape Website" button
  - Monitor progress
  - Check logs

### ☑️ Performance Tests

- [ ] **Monitor GPU usage**
  ```bash
  watch -n 1 nvidia-smi
  ```
  Expected: 60-80% GPU utilization during queries

- [ ] **Check response times**
  ```bash
  tail -f logs/production_trace.log | grep "Ollama response"
  ```
  Expected: 2-8 seconds

- [ ] **Test concurrent requests**
  ```bash
  # Use Apache Bench or similar
  ab -n 100 -c 10 -p query.json -T application/json http://localhost:8000/api/chat
  ```
  Expected: Handle 10 concurrent without errors

### ☑️ Voice Tests (Optional)

- [ ] **Test voice endpoint**
  - Upload audio file via Postman/curl
  - Verify transcription
  - Verify response
  - Verify TTS audio

---

## 📅 PHASE 5: PRODUCTION DEPLOYMENT (Day 4)

### ☑️ Security

- [ ] **Change admin password**
  ```bash
  curl -X PUT http://localhost:8000/api/admin/users/change-password \
    -H "Content-Type: application/json" \
    -d '{"username": "admin", "password": "NEW_STRONG_PASSWORD"}'
  ```

- [ ] **Add new admin users** (if needed)
  ```bash
  curl -X POST http://localhost:8000/api/admin/users/add \
    -H "Content-Type: application/json" \
    -d '{"username": "new_admin", "password": "STRONG_PASSWORD"}'
  ```

- [ ] **Setup firewall**
  ```bash
  sudo ufw allow 8000/tcp
  sudo ufw allow 80/tcp
  sudo ufw allow 443/tcp
  sudo ufw enable
  ```

### ☑️ Systemd Service

- [ ] **Create service file**
  ```bash
  sudo nano /etc/systemd/system/goeic-rag.service
  ```

- [ ] **Configure service**
  See SETUP_GUIDE.md for template

- [ ] **Enable and start**
  ```bash
  sudo systemctl daemon-reload
  sudo systemctl enable goeic-rag
  sudo systemctl start goeic-rag
  ```

- [ ] **Verify service**
  ```bash
  sudo systemctl status goeic-rag
  ```

### ☑️ Nginx Setup

- [ ] **Install Nginx**
  ```bash
  sudo apt install nginx -y
  ```

- [ ] **Configure reverse proxy**
  See SETUP_GUIDE.md for configuration

- [ ] **Enable site**
  ```bash
  sudo ln -s /etc/nginx/sites-available/goeic-rag /etc/nginx/sites-enabled/
  sudo nginx -t
  sudo systemctl restart nginx
  ```

- [ ] **Setup SSL (optional)**
  ```bash
  sudo apt install certbot python3-certbot-nginx
  sudo certbot --nginx -d your_domain.com
  ```

### ☑️ Monitoring

- [ ] **Setup log rotation**
  ```bash
  sudo nano /etc/logrotate.d/goeic-rag
  ```
  
  ```
  /home/user/goeic_rag/logs/*.log {
      daily
      rotate 7
      compress
      delaycompress
      notifempty
      create 0640 user user
  }
  ```

- [ ] **Setup GPU monitoring script**
  ```bash
  # Create monitoring script
  cat > monitor_gpu.sh << 'EOF'
  #!/bin/bash
  while true; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader >> gpu_monitor.log
    sleep 60
  done
  EOF
  
  chmod +x monitor_gpu.sh
  nohup ./monitor_gpu.sh &
  ```

---

## 📅 PHASE 6: VALIDATION & HANDOFF (Day 5)

### ☑️ Final Checks

- [ ] **All services running**
  ```bash
  sudo systemctl status goeic-rag
  sudo systemctl status nginx
  docker ps | grep weaviate
  ps aux | grep ollama
  ```

- [ ] **Database populated**
  ```bash
  curl http://localhost:8000/api/admin/database-stats
  ```
  Expected: Thousands of documents

- [ ] **Logs clean**
  ```bash
  tail -100 logs/production_trace.log | grep ERROR
  ```
  Expected: No critical errors

- [ ] **Performance acceptable**
  - Response times: 2-8 seconds ✅
  - GPU usage: 60-80% ✅
  - No memory leaks ✅

### ☑️ Documentation

- [ ] **Document configuration**
  - .env settings
  - Systemd service config
  - Nginx config
  - Admin credentials (secure location!)

- [ ] **Create operational runbook**
  - How to restart service
  - How to check logs
  - How to trigger re-scraping
  - How to add users

- [ ] **Train administrators**
  - Show admin panel
  - Explain scraping process
  - Show monitoring tools

### ☑️ Backup & Recovery

- [ ] **Setup automated backups**
  ```bash
  # Create backup script
  cat > backup.sh << 'EOF'
  #!/bin/bash
  DATE=$(date +%Y%m%d)
  docker exec weaviate sh -c 'tar -czf - /var/lib/weaviate' > backups/weaviate_$DATE.tar.gz
  tar -czf backups/logs_$DATE.tar.gz logs/
  EOF
  
  chmod +x backup.sh
  
  # Add to crontab (daily 2 AM)
  (crontab -l 2>/dev/null; echo "0 2 * * * /path/to/backup.sh") | crontab -
  ```

- [ ] **Test restore procedure**
  ```bash
  # Stop service
  sudo systemctl stop goeic-rag
  docker-compose down
  
  # Restore backup
  tar -xzf backups/weaviate_20240101.tar.gz
  
  # Restart
  docker-compose up -d
  sudo systemctl start goeic-rag
  ```

---

## 📅 PHASE 7: DECOMMISSION OLD SYSTEM (Day 6+)

### ☑️ Parallel Running (1 week)

- [ ] **Run both systems in parallel**
- [ ] **Monitor both for comparison**
- [ ] **Gather user feedback**
- [ ] **Compare costs** (should see $0 on new system!)

### ☑️ Final Migration

- [ ] **Point DNS to new system** (if applicable)
- [ ] **Update documentation**
- [ ] **Notify users of migration**

### ☑️ Cleanup

- [ ] **Stop old system**
  ```bash
  # On old server
  sudo systemctl stop old-goeic-service
  ```

- [ ] **Archive old system**
  ```bash
  tar -czf old_system_archive.tar.gz /path/to/old/system
  mv old_system_archive.tar.gz /backups/
  ```

- [ ] **Cancel API subscriptions**
  - [ ] Cancel OpenAI subscription
  - [ ] Cancel Google AI subscription
  - [ ] Verify no charges

---

## ✅ SUCCESS CRITERIA

Migration is successful when:

- ✅ Application running stable for 7 days
- ✅ Response quality 90%+ of old system
- ✅ Response times acceptable (2-8 seconds)
- ✅ No API costs (verify billing = $0)
- ✅ Admin panel working (scraping, uploads)
- ✅ Voice features working (if used)
- ✅ Analytics working
- ✅ All 3 languages working (AR/EN/FR)
- ✅ No duplicate entries in database
- ✅ Backups configured and tested
- ✅ Team trained on new system

---

## 📊 ROLLBACK PLAN

If migration fails:

1. **Keep old system running** during migration
2. **DNS/traffic** can be switched back instantly
3. **Weaviate backup** can be restored in 10 minutes
4. **Old API keys** kept active for 1 month

**Rollback decision point:** End of Phase 6 (Day 5)

---

## 💡 TIPS FOR SUCCESS

1. **Don't rush** - Take time to test thoroughly
2. **Monitor GPU** - Should be 60-80% utilized
3. **Check logs frequently** - Catch issues early
4. **Compare quality** - Test same queries on both systems
5. **Backup everything** - Before each major step
6. **Document changes** - What worked, what didn't

---

## 🎉 CONGRATULATIONS!

Once you complete this checklist, you'll have:

✅ **$0 monthly costs** (instead of $2,000-4,000)  
✅ **100% offline** (no API dependencies)  
✅ **Better control** (duplicate prevention, background tasks)  
✅ **Same quality** (90-95% of GPT-4)  
✅ **Enhanced admin panel**  
✅ **Privacy** (all data local)  

**Estimated total migration time: 4-6 days**  
**Estimated annual savings: $24,000 - $48,000**
