#!/bin/bash
echo "Testing classify_deviation..."
AUDITRX_TASK=classify_deviation python inference.py
echo "Testing draft_capa..."
AUDITRX_TASK=draft_capa python inference.py
echo "Testing negotiate_escalation..."
AUDITRX_TASK=negotiate_escalation python inference.py
