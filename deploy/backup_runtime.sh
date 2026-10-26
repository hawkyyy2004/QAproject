#!/bin/bash
# QAproject runtime daily backup, keep 14 days
# crontab registration (one-time):
#   0 2 * * * /home/public/qaproject/deploy/backup_runtime.sh >> /home/public/backups/backup.log 2>&1
BACKUP_DIR=/home/public/backups
mkdir -p "$BACKUP_DIR"
tar -czf "$BACKUP_DIR/qa-runtime-$(date +%F).tar.gz" -C /home/public/qaproject/runtime live
find "$BACKUP_DIR" -name 'qa-runtime-*.tar.gz' -mtime +14 -delete