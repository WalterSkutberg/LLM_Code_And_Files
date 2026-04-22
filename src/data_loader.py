"""
Datainsamling, rensning och förberedelse
"""
import os
import re
import requests
from pathlib import Path
from typing import List, Tuple
import pickle


class DataLoader:
    """
    Samlar in, rensar och förbereder textdata för träning.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Skapa kataloger
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_frankenstein(self, save_path: str = None) -> str:
        """
        Ladda ner Mary Shelley's Frankenstein från Project Gutenberg.
        
        Returns:
            str: Sökväg till sparad fil
        """
        if save_path is None:
            save_path = str(self.raw_dir / "frankenstein.txt")
        
        # Frankenstein från Project Gutenberg (UTF-8)
        url = "https://www.gutenberg.org/cache/epub/84/pg84.txt"
        
        print(f"Laddar ner Frankenstein från Project Gutenberg...")
        try:
            response = requests.get(url, timeout=10)
            response.encoding = 'utf-8'
            text = response.text
            
            # Spara till fil
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"✓ Sparad: {save_path}")
            return save_path
        except Exception as e:
            print(f"✗ Fel vid nedladdning: {e}")
            return None
    
    def clean_text(self, text: str) -> str:
        """
        Rensar text från:
        - Project Gutenberg metadata
        - Dubbletter
        - Felaktig formatering
        - Personuppgifter
        
        Args:
            text (str): Rå text
            
        Returns:
            str: Rensat text
        """
        # Ta bort Project Gutenberg header/footer
        # Hitta "*** START ***" och "*** END ***"
        start_markers = [
            "*** START",
            "***START",
            "FRANKENSTEIN"
        ]
        end_markers = [
            "*** END",
            "***END",
            "End of Project Gutenberg"
        ]
        
        for marker in start_markers:
            if marker in text:
                text = text[text.find(marker):]
                # Hitta slutet av raden
                text = text[text.find('\n')+1:]
                break
        
        for marker in end_markers:
            if marker in text:
                text = text[:text.find(marker)]
                break
        
        # Ta bort många mellanslag på rad
        text = re.sub(r'\s+', ' ', text)
        
        # Ta bort likstreck och andra formatering
        text = re.sub(r'_+', '', text)
        text = re.sub(r'\*+', '', text)
        
        # Dela in i meningar med radbrytningar
        # Hitta meningarna och lägg till radbrytningar
        text = re.sub(r'(?<=[.!?])\s+', '\n', text)
        
        return text.strip()
    
    def remove_duplicates(self, text: str) -> str:
        """
        Tar bort duplikatmeningar.
        
        Args:
            text (str): Text
            
        Returns:
            str: Text utan dubbletter
        """
        lines = text.split('\n')
        seen = set()
        unique_lines = []
        
        for line in lines:
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                unique_lines.append(line)
        
        return '\n'.join(unique_lines)
    
    def remove_personal_data(self, text: str) -> str:
        """
        Försöker ta bort möjlig personuppgift (email, telefon, etc).
        
        Args:
            text (str): Text
            
        Returns:
            str: Text utan identifierbar personuppgift
        """
        # Ta bort email
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Ta bort telefonnummer
        text = re.sub(r'\b\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}\b', '[PHONE]', text)
        
        # Ta bort URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        
        return text
    
    def prepare_data(self, input_file: str, output_file: str = None) -> str:
        """
        Kör genom alla rensningssteg.
        
        Args:
            input_file (str): Sökväg till rå textfil
            output_file (str): Sökväg till utgångsfil
            
        Returns:
            str: Sökväg till rensat data
        """
        if output_file is None:
            output_file = str(self.processed_dir / "cleaned_text.txt")
        
        print(f"\nRensar text från {input_file}...")
        
        # Läs raw text
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Ursprunglig storlek: {len(text)} tecken")
        
        # Rensa
        text = self.clean_text(text)
        print(f"Efter rensning: {len(text)} tecken")
        
        text = self.remove_personal_data(text)
        
        text = self.remove_duplicates(text)
        print(f"Efter borttagning av dubbletter: {len(text)} tecken")
        
        # Spara
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"✓ Rensat data sparad: {output_file}")
        return output_file
    
    def split_data(self, input_file: str, 
                   train_ratio: float = 0.85,
                   val_ratio: float = 0.10) -> Tuple[str, str, str]:
        """
        Delar upp data i träning, validering och test.
        
        Args:
            input_file (str): Sökväg till rensat data
            train_ratio (float): Andel träningsdata
            val_ratio (float): Andel valideringsdata
            
        Returns:
            Tuple[str, str, str]: Sökvägar till train, val, test filer
        """
        print(f"\nDelar upp data...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Dela upp efter tecken
        total_len = len(text)
        train_end = int(total_len * train_ratio)
        val_end = int(total_len * (train_ratio + val_ratio))
        
        train_text = text[:train_end]
        val_text = text[train_end:val_end]
        test_text = text[val_end:]
        
        # Spara
        train_file = str(self.processed_dir / "train.txt")
        val_file = str(self.processed_dir / "val.txt")
        test_file = str(self.processed_dir / "test.txt")
        
        for split_text, split_file, name in [
            (train_text, train_file, "Träning"),
            (val_text, val_file, "Validering"),
            (test_text, test_file, "Test")
        ]:
            with open(split_file, 'w', encoding='utf-8') as f:
                f.write(split_text)
            ratio = len(split_text) / total_len * 100
            print(f"✓ {name}: {len(split_text)} tecken ({ratio:.1f}%)")
        
        return train_file, val_file, test_file
    
    def get_data_info(self, file_path: str) -> dict:
        """
        Får grundläggande statistik om data.
        
        Args:
            file_path (str): Sökväg till datafil
            
        Returns:
            dict: Statistik
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        lines = text.split('\n')
        words = text.split()
        
        return {
            'characters': len(text),
            'lines': len([l for l in lines if l.strip()]),
            'words': len(words),
            'avg_line_length': sum(len(l) for l in lines) / max(len(lines), 1),
            'unique_chars': len(set(text))
        }


if __name__ == "__main__":
    # Demo
    loader = DataLoader()
    
    # Ladda ner data
    raw_file = loader.download_frankenstein()
    
    if raw_file:
        # Rensa
        clean_file = loader.prepare_data(raw_file)
        
        # Dela upp
        train_file, val_file, test_file = loader.split_data(clean_file)
        
        # Visa statistik
        print("\n=== STATISTIK ===")
        for name, file in [("Träning", train_file), ("Validering", val_file), ("Test", test_file)]:
            stats = loader.get_data_info(file)
            print(f"\n{name}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
