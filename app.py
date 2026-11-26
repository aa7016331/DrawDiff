import os
import csv
import io
import base64
import json
from pathlib import Path
from collections import Counter

import cv2
import fitz
import numpy as np
from flask import Flask, render_template_string, request, jsonify, Response


BASE = Path(__file__).resolve().parent
SYMBOL_DIR = BASE / "static" / "symbols"

# 內嵌本版本專用的 13 張符號範本，避免沿用舊資料夾中的不同版本。
_EMBEDDED_SYMBOL_PNG = {
    0: (
        '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'
        'RU5ErkJggg=='
    ),
    1: (
        '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'
        '0H8R1lvW/kPCjAAAAABJRU5ErkJggg=='
    ),
    2: (
        'iVBORw0KGgoAAAANSUhEUgAAAEQAAAAdCAIAAACc8F3aAAAGCElEQVR4nL1YvWsUQRR/M7O7dxERC0VEMFgLKpy5GAxBxM4+hY2I'
        'WigiYvwH7EQLi4CgXSoLIeEqEYMQbAQ1gtFIDGpMEOGExM+92/l6Fj9vXE+JexfxcRxvZmdn3++9N+9jBDNTt+S9F0KADwwRWWuV'
        'UmEGy/IL1kjOeBVJIQUzG2OIKI5ja61YCxhQm6zeeyklGPAY/ktiIiIhRaPRKJfLRKS1TpIkWsueeVm998yMoXNOKdUGw1obRWv6'
        'XCDnHTOXy2WlFDN775MkIaI1WQZgnHP1ev3Nmzf1et0YE0XR+vXrd+zY0dvbG8cxQCql/gmMPAkhgERKCRRFHcA5B8ZaG5BkWTYx'
        'MXH48OHh4eGpqSlr7fDwsHNufn7+/PnzQ0ND165dW1lZUUrhdXwy7LB2METUMr4QQhAXJmMMMzvn8D89Pb13796RkZGFhYVms6m1'
        'hqxYrLVeXFwcHR0lovHxcbzFzNba8B92645gGZD3zMyFwDjnGo1GXoixsbFqtTozM4N5QPXew4ZBSmvt0tLS6dOnz549m6ZpfnHY'
        'GX7YEXlm3w7GFwUTZGVmrfXo6OiJEyc+fPjQbDaDTGBwIoE5PF1eXr5x48bx48eZudFoYJ9g5/8NJojFzPfu3Ttw4EC9XsewTe7g'
        'ZvhA3ouOHTt26dIlwMBbXfsYwJAUQQ2dWQZyf/z4cf/+/c+ePYP0eY+ClEop2BBHiJk/f/4MJsuy/v7+hYUF8Jxztv8KJkh269at'
        '69evgwceZMYACZYJRwsvWmsB7+7du2fOnAnYuFvj5MH8+HVqGWbu7+9//fp18PIAA0Ij/obJcB7AY/3AwMDi4mJeHd3ZJ2cZx/zj'
        'Q0XzDDPPzc3t2rVr+/btCPDOOTBCiDiOEfJDXYP0n+exyZEjR6anpwE7iiKtNcoCbqWgkNBWlYZadZ7HjwVxwaTJzEKIJ0+eDA0N'
        'KaWQLrtI6lLKwcHBBw8eZFlGREoplCFBL0opxEMsWAUMyjOPAROx8AXBCCG8958+fdq6dSszSym7q7Kcc5s3b56bm1u3bl2Y0VqH'
        'BAXPNMaUSqUC+0VEklgEFEXdTEo5Ozu7bds2RHfor1MwSqktW7ZMTExg2Gg0YJw4jnGioiiSUsJcqxlHwDiiBaETMNBcT08PmgcM'
        'uwATCmetNRGVy2VYg3IHL0SR1YzDRD+bI0lMgnOYVicEkIGBgZcvXxIRkkkXXUoURTMzMxcuXEiSRAgRRVGpVEqSBKEiSRLvfdgc'
        'Kvsz/d7mMREXA4O2cdOmTS9evAjYOkUCqtfrvb292PPr16/W2jRNkXOazSYRee/jOHbO/V1Zol2GotrNsmznzp21Wi1NU2ixUAz9'
        'jW7evLlnzx4kn56eHq11qVSC9AgtRTUliIT9ZVjczUql0oYNGwYHB58+fYpJhGatNaKq995ai7MbpMk3P0T05cuXR48eVSoVvMLM'
        'cC1qBWVqecGqcd+T8ESWiIksCfvT64qkW5Qn3vu3b9/u3r07eAWeGmNCvUOtFiCf2pGevffnzp2r1WqYjKKoeL7/lRyzYdYUkWXN'
        'rNk779j7TsoZJIGRkZGrV68GkKE1ALwkSYL0gbTWaZo+fPiwWq1mWQYt/HSQXCXfLRjjHfvitVm+aO/r67t9+3ZbTZXvNAEGBgS9'
        'e/eOiObn58NbwYad9zMOeLoEE7yImdM0XVpa2rdv39jYWFAzt5xKCGGMaauFJycnoyi6f/9+UAcSIp520ZwxmmRFlh3AOM+2uGVC'
        'WwKB6vX6yZMnT5069erVK1T4mIebBXgrKytXrlzp6+ubnZ0NSH6U60TGmHzb1xkYz6SUZWY27F0HYCBEuAaAEYwx4+PjlUrl6NGj'
        'k5OT79+/R2Rzzi0vL09NTV28eFFKefny5XypjyKVmZEWOdf8dECO2TPJxGLQAlPo3gwpDCc1RFUw3759e/78+Z07dx4/flyr1SBi'
        'pVI5dOjQwYMHq9Xqxo0b/7hn/tbrrwK0ExMTSVWyPlNMJDyzJLG2S0BQuIbL3yZnWYbiChenYSW3LgRDEOviDtozEZFSsTE6UsJb'
        'R1JISd8BZgSg2f9PSeMAAAAASUVORK5CYII='
    ),
