from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from src.ingestion.loader import SourceFile


@dataclass
class CodeChunk:
    """One chunk of code, with enough metadata to cite it later."""
    file_path: str       
    content: str          
    start_line: int       
    end_line: int         


def chunk_source_file(source_file: SourceFile, chunk_size: int = 500, chunk_overlap: int = 50) -> list[CodeChunk]:
    """
    Split one file into CodeChunks, splitting on Python syntax boundaries
    where possible (def/class) instead of blindly every N characters.
    """
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    raw_chunks = splitter.split_text(source_file.content)

    
    chunks = []
    search_start_pos = 0  

    for raw_chunk_text in raw_chunks:
        
        char_pos = source_file.content.find(raw_chunk_text, search_start_pos)
        if char_pos == -1:
            char_pos = source_file.content.find(raw_chunk_text)

        
        start_line = source_file.content.count("\n", 0, char_pos) + 1
        end_line = start_line + raw_chunk_text.count("\n")

        chunks.append(CodeChunk(
            file_path=source_file.file_path,
            content=raw_chunk_text,
            start_line=start_line,
            end_line=end_line,
        ))

        search_start_pos = char_pos + 1  

    return chunks


def chunk_all_files(source_files: list[SourceFile]) -> list[CodeChunk]:
    """Convenience wrapper: chunk every file in a list, return one flat list of chunks."""
    all_chunks = []
    for sf in source_files:
        all_chunks.extend(chunk_source_file(sf))
    return all_chunks


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../..")
    from src.ingestion.loader import load_repo_files

    test_path = sys.argv[1] if len(sys.argv) > 1 else "../../repos/micrograd"
    files = load_repo_files(test_path)
    chunks = chunk_all_files(files)

    print(f"Produced {len(chunks)} chunks from {len(files)} files:\n")
    for c in chunks:
        preview = c.content.strip().replace("\n", " \\n ")[:70]
        print(f"  [{c.file_path}:{c.start_line}-{c.end_line}]  {preview}...")