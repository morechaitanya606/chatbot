import ray
import logging
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import faiss  # Import FAISS from the actual library
import numpy as np  # Make sure numpy is imported for array handling

# Initialize Ray
ray.init()

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load documents with logging
logging.info("Loading documents...")
loader = DirectoryLoader('data', glob="./*.txt")
documents = loader.load()

# Extract text from documents and split into manageable texts with logging
logging.info("Extracting and splitting texts from documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
texts = []
for document in documents:
    if hasattr(document, 'get_text'):
        text_content = document.get_text()  # Adjust according to actual method
    else:
        text_content = ""  # Default to empty string if no text method is available

    texts.extend(text_splitter.split_text(text_content))

# Initialize HuggingFaceEmbeddings directly
embeddings_model = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")

# Create FAISS index for embeddings
index = faiss.IndexFlatL2(768)  # Dimension of embeddings, adjust as needed

# Assuming docstore as a simple dictionary to store document texts
docstore = {i: text for i, text in enumerate(texts)}
index_to_docstore_id = {i: i for i in range(len(texts))}

# Initialize FAISS with the embeddings model
faiss_db = FAISS(embeddings_model, index, docstore, index_to_docstore_id)

# Process and store embeddings in batches
batch_size = 526  # Adjust batch size depending on available memory
batch_embeddings = []
batch_texts = []

logging.info("Storing embeddings in FAISS...")
for i, text in enumerate(texts):
    embedding = embeddings_model.embed_query(text)

    # Debug: Check if embedding is a numpy array and its shape
    if isinstance(embedding, np.ndarray):
        logging.debug(f"Embedding {i} shape: {embedding.shape}")
    else:
        logging.warning(f"Embedding {i} is not a valid numpy array. Skipping.")
        continue

    # Add the embedding to the batch
    batch_embeddings.append(embedding)
    batch_texts.append(text)
    
    # When batch size is reached or it's the last batch, add to FAISS
    if len(batch_embeddings) >= batch_size or i == len(texts) - 1:
        logging.info(f"Adding batch {i//batch_size + 1} of embeddings...")
        
        # Convert the batch of embeddings into a numpy array before adding to FAISS
        batch_embeddings_array = np.array(batch_embeddings)

        # Log the shape of the batch embeddings to ensure correct dimensions
        logging.debug(f"Batch shape: {batch_embeddings_array.shape}")

        if batch_embeddings_array.shape[0] > 0:  # Check if the batch has data
            faiss_db.add_documents(batch_embeddings_array)  # Add batch to FAISS
            logging.info(f"Added {len(batch_embeddings_array)} embeddings to FAISS.")
        
        batch_embeddings = []  # Reset the batch for the next set of embeddings
import numpy as np
import faiss

# Create a small sample embedding
embeddings = np.random.random((10, 768)).astype('float32')

# Create a FAISS index
index = faiss.IndexFlatL2(768)
index.add(embeddings)

# Check if the index size increased
print(f"FAISS index size: {index.ntotal}")

# Exporting the vector embeddings database with logging
logging.info("Exporting the vector embeddings database...")
faiss_db.save_local("ipc_embed_db")

# Log a message to indicate the completion of the process
logging.info("Process completed successfully.")

# Shutdown Ray after the process
ray.shutdown()
