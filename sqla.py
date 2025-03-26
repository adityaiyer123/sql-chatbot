import json
import re
import sqlparse
import logging
import sys
from datetime import datetime
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import oracledb
import uvicorn
from langchain_community.embeddings.oci_generative_ai import OCIGenAIEmbeddings
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.vectorstores.oraclevs import OracleVS, DistanceStrategy
from langchain.schema import Document

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = FastAPI()

ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
EMBEDDING_MODEL = "cohere.embed-english-v3.0"
GENERATE_MODEL = "meta.llama-3.3-70b-instruct"
ORACLE_COMPARTMENT_ID = "ocid1.tenancy.oc1..aaaaaaaaweuxa6ovnhihlbpolrh3jrdpasnnukjd5x5slxcekzwsdigsayza"

DB_USER = "admin"
DB_PWD = "DMCCDubai@2020"
WALLET_DIR = r"C:\Users\AdityaIyer\Downloads\Wallet_DMCCDEVATP"
WALLET_PWD = "DMCCDubai@2020"
DSN = "(description= (retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1522)(host=adb.me-dubai-1.oraclecloud.com))(connect_data=(service_name=g84283fcf3e7fe8_dmccdevatp_high.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))"
SCHEMA = "ADMIN"
TABLES = ["INVPAYMENT2AI", "INVOICEGMAI"]
TABLE_NAME = "AI_SQL_BOT2_EMBEDDINGS"

embedding_model = OCIGenAIEmbeddings(
    model_id=EMBEDDING_MODEL,
    service_endpoint=ENDPOINT,
    compartment_id=ORACLE_COMPARTMENT_ID
)

def initialize_llm():
    """Initialize OCI Generative AI LLM"""
    try:
        logger.debug("Initializing OCI GenAI LLM")
        llm_instance = ChatOCIGenAI(
            model_id=GENERATE_MODEL,
            service_endpoint=ENDPOINT,
            compartment_id=ORACLE_COMPARTMENT_ID,
            model_kwargs={"temperature": 0, "max_tokens": 3000},
            auth_type="API_KEY"
        )
        logger.debug("LLM initialized successfully")
        return llm_instance
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        raise HTTPException(status_code=500, detail="LLM Initialization Failed")

llm = initialize_llm()

def get_column_names() -> Dict[str, List[str]]:
    """Retrieve column names for only the specified tables from Oracle DB"""
    connection = oracledb.connect(
        user=DB_USER, password=DB_PWD, dsn=DSN,
        wallet_location=WALLET_DIR, wallet_password=WALLET_PWD
    )
    cursor = connection.cursor()
    table_columns = {}
    for table_name in TABLES:
        cursor.execute(
            f"SELECT column_name FROM ALL_TAB_COLUMNS WHERE table_name = '{table_name}' AND owner = '{SCHEMA.upper()}'"
        )
        table_columns[table_name] = [row[0] for row in cursor.fetchall()]
    cursor.close()
    connection.close()
    return table_columns

def generate_column_descriptions() -> Dict[str, Dict[str, str]]:
    """
    Generate detailed column descriptions for each specified table based on column names and their data types.
    Uses the LLM to create human-friendly descriptions.
    Returns a dictionary mapping table names to a dictionary of column descriptions.
    """
    try:
        connection = oracledb.connect(
            user=DB_USER, password=DB_PWD, dsn=DSN,
            wallet_location=WALLET_DIR, wallet_password=WALLET_PWD
        )
        cursor = connection.cursor()
        table_descriptions = {}
        for table_name in TABLES:
            query = f"SELECT column_name, data_type FROM ALL_TAB_COLUMNS WHERE table_name = '{table_name}' AND owner = '{SCHEMA.upper()}'"
            cursor.execute(query)
            columns_info_all = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor.execute(f"SELECT * FROM {SCHEMA.upper()}.{table_name} WHERE 1=0")
            columns_info_describe = {}
            for col in cursor.description:
                col_name = col[0]
                col_type = getattr(col[1], '__name__', str(col[1]))
                columns_info_describe[col_name] = col_type

            prompt = (
                f"Given the following columns and their Oracle data types for the table '{table_name}':\n"
                f"From ALL_TAB_COLUMNS:\n{json.dumps(columns_info_all, indent=4)}\n\n"
                f"From DESCRIBE (simulated via SELECT * FROM {SCHEMA.upper()}.{table_name} WHERE 1=0):\n{json.dumps(columns_info_describe, indent=4)}\n\n"
                "Provide a detailed description for each column. The description should explain the type of data stored in the column "
                "and its significance. Return the result as a JSON object where the keys are the column names and the values are the corresponding detailed descriptions."
            )
            llm_response = llm.invoke(prompt)
            try:
                descriptions = json.loads(llm_response.content.strip())
            except json.JSONDecodeError:
                descriptions = {"description": llm_response.content.strip()}
            table_descriptions[table_name] = descriptions
        cursor.close()
        connection.close()
        return table_descriptions
    except Exception as e:
        logger.error(f"Error generating column descriptions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating column descriptions: {str(e)}")

def format_sql_for_oracle(sql_query: str) -> str:
    """Format SQL query for Oracle compatibility"""
    sql_query = sql_query.strip().rstrip(";")
    sql_query = re.sub(r"LIMIT (\d+)", r"FETCH FIRST \\1 ROWS ONLY", sql_query, flags=re.IGNORECASE)
    return sqlparse.format(sql_query, reindent=True, keyword_case="upper")

def convert_datetime_in_results(results: List[Dict]) -> List[Dict]:
    """Convert datetime objects in result dictionaries to strings."""
    for row in results:
        for key, value in row.items():
            if isinstance(value, datetime):
                row[key] = value.strftime("%Y-%m-%d %H:%M:%S")
    return results

@app.post("/ingest-documents/")
async def ingest_documents():
    """
    Connect to Oracle DB, fetch data from INVPAYMENT2AI and INVOICEGMAI tables,
    transform each row into a JSON record, and ingest them into Oracle 23AI Vector DB.
    """
    documents: List[Document] = []
    try:
        connection = oracledb.connect(
            user=DB_USER,
            password=DB_PWD,
            dsn=DSN,
            wallet_location=WALLET_DIR,
            wallet_password=WALLET_PWD
        )
        logger.info("Oracle connection successful!")
    except oracledb.Error as e:
        raise HTTPException(status_code=500, detail=f"Oracle connection error: {str(e)}")
    try:
        cursor = connection.cursor()
        sql_payment = (
            "SELECT BU_NAME, PO_NUMBER, SUPPLIER_NAME, SUPPLIER_NUMBER, "
            "PO_DATE, PO_AMOUNT, PO_STATUS, REQUISTION_NO, REQUISITION_STATUS, "
            "INVOICE_NUM, INVOICE_DATE, INVOICE_STATUS, INVOICE_AMOUNT, CURRENCY, "
            "PAYMENT_NUMBER, PAYMENT_DATE, PAID_AMOUNT "
            "FROM INVPAYMENT2AI"
        )
        cursor.execute(sql_payment)
        rows_payment = cursor.fetchall()
        logger.info(f"Fetched {len(rows_payment)} rows from INVPAYMENT2AI.")
        for row in rows_payment:
            record = {
                "BU_NAME": row[0],
                "PO_NUMBER": row[1],
                "SUPPLIER_NAME": row[2],
                "SUPPLIER_NUMBER": row[3],
                "PO_DATE": str(row[4]) if row[4] else None,
                "PO_AMOUNT": row[5],
                "PO_STATUS": row[6],
                "REQUISTION_NO": row[7],
                "REQUISITION_STATUS": row[8],
                "INVOICE_NUM": row[9],
                "INVOICE_DATE": str(row[10]) if row[10] else None,
                "INVOICE_STATUS": row[11],
                "INVOICE_AMOUNT": row[12],
                "CURRENCY": row[13],
                "PAYMENT_NUMBER": row[14],
                "PAYMENT_DATE": str(row[15]) if row[15] else None,
                "PAID_AMOUNT": row[16],
                "source_table": "INVPAYMENT2AI"
            }
            json_record = json.dumps(record)
            doc = Document(page_content=json_record, metadata=record)
            documents.append(doc)
        sql_invoice = (
            "SELECT INVOICE_NUM, GRN_NUMBER, INVOICE_LINE_NUMBER, GRN_STATUS, "
            "CURRENCY, SUPPLIER_NAME, SUPPLIER_NUMBER, BU_NAME "
            "FROM INVOICEGMAI"
        )
        cursor.execute(sql_invoice)
        rows_invoice = cursor.fetchall()
        logger.info(f"Fetched {len(rows_invoice)} rows from INVOICEGMAI.")
        for row in rows_invoice:
            record = {
                "INVOICE_NUM": row[0],
                "GRN_NUMBER": row[1],
                "INVOICE_LINE_NUMBER": row[2],
                "GRN_STATUS": row[3],
                "CURRENCY": row[4],
                "SUPPLIER_NAME": row[5],
                "SUPPLIER_NUMBER": row[6],
                "BU_NAME": row[7],
                "source_table": "INVOICEGMAI"
            }
            json_record = json.dumps(record)
            doc = Document(page_content=json_record, metadata=record)
            documents.append(doc)
        cursor.close()
        if not documents:
            connection.close()
            raise HTTPException(status_code=404, detail="No documents found in the tables.")
        vector_store = OracleVS.from_documents(
            documents=documents,
            embedding=embedding_model,
            client=connection,
            table_name=TABLE_NAME,
            distance_strategy=DistanceStrategy.DOT_PRODUCT
        )
        connection.close()
        return {"message": f"Successfully ingested {len(documents)} documents into Oracle Vector Store."}
    except Exception as e:
        connection.close()
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")

class QueryRequest(BaseModel):
    user_query: str

@app.post("/query/")
async def query_database(query_request: QueryRequest):
    """
    Handle user queries by:
      1. Converting the user query to an embedding.
      2. Performing a cosine similarity search against the stored vectors.
      3. Using the retrieved (relevant) documents as context for the LLM to generate a fully executable SQL query.
      4. Verifying, executing the SQL, and returning the results along with a final answer from the LLM.
    """
    original_query = query_request.user_query
    logger.debug("Original user query: " + original_query)
    try:
        query_embedding = embedding_model.embed_query(original_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating query embedding: {str(e)}")
    try:
        connection = oracledb.connect(
            user=DB_USER,
            password=DB_PWD,
            dsn=DSN,
            wallet_location=WALLET_DIR,
            wallet_password=WALLET_PWD
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Oracle connection error: {str(e)}")
    try:
        vector_store = OracleVS(
            client=connection,
            table_name=TABLE_NAME,
            distance_strategy=DistanceStrategy.COSINE,
            embedding_function=embedding_model.embed_query
        )
        relevant_docs = vector_store.similarity_search_by_vector(query_embedding, k=25)
        connection.close()
    except Exception as e:
        connection.close()
        raise HTTPException(status_code=500, detail=f"Error during similarity search: {str(e)}")
    if not relevant_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found based on the query.")
    docs_context = [json.loads(doc.page_content) for doc in relevant_docs]
    table_columns = get_column_names()
    column_descriptions = generate_column_descriptions()
    
    # Candidate prompt: Instruct LLM to choose from various date filtering methods.
    prompt = f"""
Based on the user's query:
"{original_query}"

Below is the relevant data extracted via cosine similarity search (literal values exactly as stored):
{json.dumps(docs_context, indent=4)}

Generate a fully executable SQL query for Oracle Database based on the user query and the relevant data.
Use only the following tables: {', '.join(TABLES)}.
Use the following schema:
{json.dumps(table_columns, indent=4)}

Column Descriptions:
{json.dumps(column_descriptions, indent=4)}

Note:
- Use the column names exactly as they exist in the tables.
- If the user query references a year or a date (for example, "2025","2024", "this year", or "last year"), examine the column descriptions and choose a column whose data type is DATE or TIMESTAMP for filtering.
- You may use any appropriate date filtering method (such as EXTRACT, TRUNC, BETWEEN, or date arithmetic) based on the context.
- For relative phrases like "last year", you can use an expression such as:
    EXTRACT(YEAR FROM column_name) = EXTRACT(YEAR FROM SYSTIMESTAMP) - 1
  or any equivalent method.
- Do not use any string column (e.g. INVOICE_NUM) for date filtering.
- When generating literal values for filtering, refer to the provided column descriptions (which include the column data types) and format each literal accordingly—for example, if a column is of a string type, enclose the literal in single quotes; if it is numeric or of a date type, do not enclose it in quotes.
- Format literal values correctly (strings in single quotes, dates as needed).
- Do not use unnecessary joins or UNION operations; use only the tables and columns needed to answer the query.
- Avoid using any JOIN or UNION operations unless they are absolutely necessary to answer the query. Use only the tables and columns explicitly required by the user query. 

Return only the complete SQL query (which must start with SELECT) as plain text.
Do not include any extra commentary.
Do not add ';' at the end of the SQL query.
"""
    logger.debug("LLM Prompt:\n" + prompt)
    candidate_response = llm.invoke(prompt)
    candidate_sql = candidate_response.content.strip().strip("```sql").strip("```").strip()
    logger.debug("Candidate SQL response: " + candidate_sql)
    
    # Updated verification prompt: Require only the final corrected SQL query.
    verification_prompt = f"""
You generated the following SQL query:
{candidate_sql}

Verify and, if necessary, correct the query such that:
1. All date or year filters are applied only on columns with data types DATE or TIMESTAMP.
2. No string columns (e.g., INVOICE_NUM) are used for date filtering.
3. The query may use any appropriate method (EXTRACT, TRUNC, BETWEEN, or date arithmetic) for date filtering.
4. For relative phrases like "last year", an expression such as:
   EXTRACT(YEAR FROM column_name) = EXTRACT(YEAR FROM SYSTIMESTAMP) - 1
   or an equivalent method is used.
5. Literal values are formatted correctly based on their data types. (For example, if a column is of a string type, the literal must be enclosed in single quotes.)
6.Only the necessary tables and columns are used to answer the query (avoid unnecessary joins or unions).
7. Ensure that the final SQL query does not include any unnecessary JOIN or UNION operations; only include them if they are essential to satisfy the query's requirements.
Return only the final, corrected SQL query without any additional text.
"""
    verification_response = llm.invoke(verification_prompt)
    final_sql = verification_response.content.strip().strip("```sql").strip("```").strip()
    logger.debug("Final SQL after verification: " + final_sql)
    if not final_sql.lower().startswith("select"):
        raise HTTPException(status_code=500, detail=f"Generated SQL is invalid: {final_sql}")
    formatted_sql = format_sql_for_oracle(final_sql)
    logger.debug("Formatted SQL query: " + formatted_sql)
    print("DEBUG: Formatted SQL query:\n", formatted_sql)
    sys.stdout.flush()
    try:
        connection = oracledb.connect(
            user=DB_USER,
            password=DB_PWD,
            dsn=DSN,
            wallet_location=WALLET_DIR,
            wallet_password=WALLET_PWD
        )
        cursor = connection.cursor()
        try:
            cursor.execute(formatted_sql)
            rows = cursor.fetchall()
        except oracledb.DatabaseError as e:
            error_str = str(e)
            if "ORA-00904" in error_str:
                match = re.search(r'ORA-00904: "([^"]+)"', error_str)
                invalid_column = match.group(1) if match else "UNKNOWN"
                table_hint = ""
                table_columns = get_column_names()
                for table_name, cols in table_columns.items():
                    if invalid_column in cols:
                        table_hint = f"The column {invalid_column} exists in table {table_name}."
                        break
                error_prompt = f"""
Your previous SQL query:
{formatted_sql}

resulted in the following Oracle error:
{error_str}

{table_hint}

Using only the following tables: {', '.join(TABLES)}.
Here is the schema:
{json.dumps(table_columns, indent=4)}

and the following column descriptions:
{json.dumps(column_descriptions, indent=4)}

Please revise the SQL query so that each column is referenced correctly and literal values are properly formatted.
Return only the final corrected SQL.
"""
                error_response = llm.invoke(error_prompt)
                corrected_sql = error_response.content.strip().strip("```sql").strip("```").strip()
                formatted_sql = format_sql_for_oracle(corrected_sql)
                cursor.execute(formatted_sql)
                rows = cursor.fetchall()
            else:
                raise HTTPException(status_code=400, detail=f"SQL Execution Error: {error_str}")
    except oracledb.DatabaseError as e:
        error_str = str(e)
        raise HTTPException(status_code=400, detail=f"SQL Execution Error: {error_str}")
    column_names = [desc[0] for desc in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]
    results = convert_datetime_in_results(results)
    final_prompt = f"""
The user asked: "{original_query}"

The executed SQL query returned the following data (with columns {column_names}):
{json.dumps(results, indent=4, default=lambda o: o.strftime("%Y-%m-%d %H:%M:%S") if isinstance(o, datetime) else o)}

Based on this data, provide a clear and concise answer to the user's query.
For example, if the question is "How many invoices have been raised by X?" and the data shows a number, then your answer should be just that number.
"""
    llm_final_response = llm.invoke(final_prompt)
    final_answer = llm_final_response.content.strip()
    logger.debug("Final answer from LLM: " + final_answer)
    cursor.close()
    connection.close()
    return {
        "query": original_query,
        "generated_sql": formatted_sql,
        "results": results,
        "final_answer": final_answer
    }

@app.post("/ingest/")
async def ingest_data():
    """Ingest data from Oracle DB into the vector store (calls the ingest-documents logic)."""
    try:
        await ingest_documents()
        return {"message": "Data ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting data: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
