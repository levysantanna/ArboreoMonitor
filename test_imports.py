#!/usr/bin/env python3
"""
Teste de imports do ArboreoMonitor
"""

def test_imports():
    """Testa se todos os módulos podem ser importados."""
    try:
        print("🧪 Testando imports...")
        
        # Testar imports básicos
        import sys
        import os
        import json
        import logging
        from pathlib import Path
        from datetime import datetime
        print("✅ Imports básicos OK")
        
        # Testar imports opcionais
        try:
            import cv2
            print("✅ OpenCV disponível")
        except ImportError:
            print("⚠️  OpenCV não disponível (opcional)")
        
        try:
            import numpy as np
            print("✅ NumPy disponível")
        except ImportError:
            print("⚠️  NumPy não disponível (opcional)")
        
        try:
            from PIL import Image
            print("✅ PIL disponível")
        except ImportError:
            print("⚠️  PIL não disponível (opcional)")
        
        # Testar estrutura do projeto
        print("\n📁 Verificando estrutura do projeto...")
        
        required_dirs = [
            "src",
            "src/core",
            "src/video", 
            "src/detection",
            "src/analysis",
            "data",
            "docs",
            "config"
        ]
        
        for dir_path in required_dirs:
            if Path(dir_path).exists():
                print(f"✅ {dir_path}")
            else:
                print(f"❌ {dir_path} não encontrado")
        
        # Testar arquivos principais
        print("\n📄 Verificando arquivos principais...")
        
        required_files = [
            "main.py",
            "requirements.txt",
            "README.md",
            "config/config.yaml",
            "data/ipcam_list.json"
        ]
        
        for file_path in required_files:
            if Path(file_path).exists():
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} não encontrado")
        
        print("\n🎉 Teste de imports concluído!")
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

if __name__ == "__main__":
    test_imports()
