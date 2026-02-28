"""
Reorganize project into proper folder structure
"""

import os
import shutil

def create_folders():
    """Create folder structure"""
    folders = [
        'core',
        'ingestion',
        'scripts',
        'docs',
        'docs/screenshots'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        # Add .gitkeep
        with open(f"{folder}/.gitkeep", 'w') as f:
            f.write("")
    
    print("✅ Folders created")

def move_files():
    """Move files to appropriate folders"""
    
    # Core files
    core_files = [
        'rag_system_pro.py',
        'llm_handler.py',
        'embeddings.py',
        'vector_store.py',
        'hybrid_retriever.py',
        'reranker.py',
        'query_rewriter.py',
        'citation_manager.py'
    ]
    
    for file in core_files:
        if os.path.exists(file):
            shutil.move(file, f'core/{file}')
            print(f"  Moved {file} → core/")
    
    # Ingestion files
    ingestion_files = [
        'ingestion_with_abstract.py',
        'extract_abstract.py'
    ]
    
    for file in ingestion_files:
        if os.path.exists(file):
            shutil.move(file, f'ingestion/{file}')
            print(f"  Moved {file} → ingestion/")
    
    # Script files
    script_files = [
        'rebuild_database_fixed.py'
    ]
    
    for file in script_files:
        if os.path.exists(file):
            shutil.move(file, f'scripts/{file}')
            print(f"  Moved {file} → scripts/")
    
    print("✅ Files moved")

def create_init_files():
    """Create __init__.py files"""
    folders = ['core', 'ingestion', 'scripts']
    
    for folder in folders:
        init_file = f"{folder}/__init__.py"
        with open(init_file, 'w') as f:
            f.write(f'"""\n{folder.capitalize()} module\n"""\n')
        print(f"  Created {init_file}")
    
    print("✅ __init__.py files created")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔄 REORGANIZING PROJECT STRUCTURE")
    print("="*60 + "\n")
    
    create_folders()
    move_files()
    create_init_files()
    
    print("\n" + "="*60)
    print("✅ REORGANIZATION COMPLETE!")
    print("="*60)
    print("\n⚠️  IMPORTANT: Now update import statements in:")
    print("   1. app.py")
    print("   2. core/rag_system_pro.py")
    print("   3. core/hybrid_retriever.py")
    print("   4. core/query_rewriter.py")
    print("   5. ingestion/ingestion_with_abstract.py")
    print("   6. scripts/rebuild_database_fixed.py")
    print("\nSee the guide for exact changes needed.")