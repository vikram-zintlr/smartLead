from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import json
from django.conf import settings
print('Initializing OpenAI API key...')
load_dotenv()

# Initialize OpenAI client
api_key = getattr(settings,'API_KEY' , None)
if not api_key:
    raise ValueError("OPEN_AI_KEY not found in environment variables")

llm = ChatOpenAI(model="gpt-4", temperature=0.0, api_key=api_key)

company_schema_template = """

1. cmp_ln_url
   - Description: LinkedIn company profile URLs
   - Type: Array of Strings (URLs)
   - Example: ["linkedin.com/company/80792554", "linkedin.com/company/zintlr"]

2. cmp_name
   - Description: Official company name
   - Type: String
   - Example: "Zintlr"

3. cmp_ind
   - Description: Industry classifications for the company
   - Type: Array of Strings categorical
   - Example: ["Business Intelligence Platforms"]

4. follow
   - Description: Number of followers on LinkedIn (formatted string)
   - Type: String
   - Example: "1,243"

5. prof_desc
    - Description: Short professional description/tagline
    - Type: String
    - Example: "Unlock Sales Intelligence: Zintlr—AI Sales Insights, Verified Contacts..."

6. cmp_ext_logo
    - Description: URL of the company logo
    - Type: String (URL)
    - Example: "https://media.licdn.com/dms/image/C560BAQEDdZOgaiYyzw/company-logo_200_200/0/1675061318659..."

7. cmp_similar_pages
    - Description: List of similar company pages on LinkedIn
    - Type: Array of Objects
    - Subkeys per object:
      - cmp_name: String (Company name)
      - cmp_ind: String (Industry)
      - cmp_loc: String (Location, optional)
      - cmp_link: String (URL to company page)
      - cmp_ext_logo: String (URL to logo)
    - Example: [{"cmp_name": "Chillitray Technologies", ...}]

8. cmp_about
    - Description: Detailed "About Us" description
    - Type: String (Long text)
    - Example: "Zintlr is a SaaS-based Sales & talent Intelligence tool..."

9. cmp_web_urls
    - Description: Company website URLs
    - Type: Array of Strings (URLs)
    - Example: ["zintlr.com"]

10. cmp_size
    - Description: Employee size range
    - Type: String categorical
    - Category:1-10,11-50,51-200,201-500,501-1k,1k-5k,5k-10k,10k+
    - Example: "11-50"

11. cmp_loc_name
    - Description: Full location name
    - Type: String
    - Example: "bengaluru, karnataka"

12. cmp_loc_zip
    - Description: Postal/ZIP code
    - Type: String
    - Example: "560001"

13-16. cmp_loc_[city|state|country|continent]_norm
    - Description: Normalized geographic names
    - Type: Strings
    - Examples: "bengaluru", "karnataka", "india", "asia"

17. cmp_loc_metro
    - Description: Metropolitan area name
    - Type: String
    - Example: "bengaluru, karnataka"


18. cmp_type
    - Description: Company ownership type
    - Type: String
    - Example: "Privately Held"

19. cmp_desc
    - Description: Extended company description
    - Type: String (Long text)
    - Example: "Zintlr is a B2B database and sales intelligence platform..."

20. ttl_follow
    - Description: Historical follower count data points
    - Type: Array of Objects
    - Subkeys per object:
      - ttl_follow: String (Follower count)
      - ttl_follow_lu: String (Timestamp when recorded)
    - Example: [{"ttl_follow": "5K", "ttl_follow_lu": "2024-08-10 22:58:13"}, ...]

21. cmp_keywords
    - Description: Relevant keywords for the company
    - Type: Array of Strings [categorical]
    - Example: ["team", "strategy", "technologie", ...]

22. cmp_tech
    - Description: Technologies used by the company
    - Type: Array of Strings
    - Example: ["Node.js", "React", "Next.js", ...]

23-24. cmp_[fb|twitter]_urls
    - Description: Social media profile URLs
    - Type: Arrays of Strings (URLs)
    - Example: [] (empty arrays in sample)

25. job_open
    - Description: List of open job positions
    - Type: Array
    - Example: [] (empty in sample)

26. cmp_fd_date
    - Description: Founding date (year)
    - Type: String
    - Example: "2022"

27. cmp_fd_yr
    - Description: Founding year (numeric)
    - Type: Integer
    - Example: 2022

28. cmp_ext_img
    - Description: URL of extended company image/logo
    - Type: String (URL)
    - Example: "https://media.licdn.com/dms/image/v2/D4D0BAQGC3fiQkYOxEA/company-logo_200_200/..."

29. ctc_info
    - Description: Contact information
    - Type: Array of Objects
    - Subkeys per object:
      - ctc_info: String (Contact details/URL)
      - ctc_info_lu: String (Last update timestamp)
    - Example: [{"ctc_info": "calendly.com/ujwal-zintlr/30min", ...}]

30. fund
    - Description: Funding information
    - Type: Array of Objects
    - Subkeys per object:
      - fund_type: String (Funding round type)
      - fund_amt: String (Funding amount)
    - Example: [{"fund_type": "Seed", "fund_amt": "US$ 895K"}]
31. cmp_revenue
    -Description;- companies revenue information
    -Type:- string categorical
    - Category:- [$0 - 1M ,$1 - 10M ,$10 - 50M ,$50 - 100M ,$100 - 250M ,$250 - 500M ,$500 - 1B ,> $1B]
32. ipo_status
    -description:- IPO Status
    - Type: string
    example:- Private
33.  ttl_funds
    - description:-  Total Number of funding rounds raised
    - Type:- string
    example:- 10
34. ttl_acqs
    - description:- Total number of companies acquired
    -Type:- string
    -Example:- 3
               
"""

print('Schema template initialized')

# Improved prompt template
prompt_template = PromptTemplate(
    input_variables=["user_query", "company_schema"],
    template="""
    You are an expert MongoDB query generator. 
    Strictly follow these rules:
    1. Use only the provided company schema
    2. Generate ONLY the MongoDB aggregation pipeline
    3. Never include explanations or additional text
    4. Format the output as valid MongoDB JSON syntax
    5. For categorical fields such as cmp_ind and cmp_keywords, if the user's query does not exactly match the schema values, generate a query that search in cmp_dec,cmp_about and prof_desc keys with case-insensitive regex matching or, if necessary, applies synonym mapping or fuzzy matching to capture similar terms.
    6. If the user requests in-depth company details and the provided filtering keys do not exactly match any keys in the schema, generate a query that returns the entire company document.
    7. Handle edge cases gracefully.   
    Company Schema: {company_schema}
    
    User Request: {user_query}
    
    MongoDB Aggregation Pipeline:
    """
)

# Create LLMChain properly
chain = LLMChain(llm=llm, prompt=prompt_template)

def generate_mongo_query(user_query):
    try:
        print("Generating query for:", user_query)
        response = chain.run({
            "user_query": user_query,
            "company_schema": company_schema_template
        })
        
        # Clean up response to ensure only JSON is returned
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]
    
        return response.strip()
    except Exception as e:
        print("Error generating MongoDB query:", str(e))
        return None

# Example usage:
query = "Find all companies in Bengaluru with more than 50 employees in the IT sector"
result = generate_mongo_query(query)
print("Generated Query:", result)