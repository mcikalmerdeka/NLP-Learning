import os
from typing import List, Optional
import requests
from anthropic import Anthropic
from datetime import datetime
import json
from dotenv import load_dotenv

# Personal reminder: Check this chat history for the source https://claude.ai/chat/e9afd21d-3e5f-4c6d-bcf1-0ad951654f44

# Load environment variables
load_dotenv()

# Define the SearchAugmentedGeneration class
class SearchAugmentedGeneration:
    def __init__(self, anthropic_api_key: str = None, serper_api_key: str = None):
        """
        Initialize the search-augmented generation system.
        
        Args:
            anthropic_api_key: API key for Anthropic's Claude
            serper_api_key: API key for Serper (Google Search API)
        """
        # Get API keys from environment variables if not provided
        self.anthropic_api_key = anthropic_api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.serper_api_key = serper_api_key or os.environ.get('SERPER_API_KEY')
        
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass it to the constructor.")
        if not self.serper_api_key:
            raise ValueError("Serper API key is required. Set SERPER_API_KEY environment variable or pass it to the constructor.")
            
        self.client = Anthropic(api_key=self.anthropic_api_key)
    
    # Define the search_web method
    def search_web(self, query: str, num_results: int = 3) -> List[dict]:
        """
        Search the web using Serper API.
        
        Args:
            query: Search query string
            num_results: Number of search results to return
            
        Returns:
            List of dictionaries containing search results
        """

        # Set headers and payload for the API request
        headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }
        
        # Set the payload for the API request
        payload = {
            'q': query,
            'num': num_results
        }
        
        # Send a POST request to the Serper API
        response = requests.post(
            'https://google.serper.dev/search',
            headers=headers,
            json=payload
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            results = response.json()
            return results.get('organic', [])
        else:
            raise Exception(f"Search failed with status code: {response.status_code}")
    
    # Define the format_search_results method
    def format_search_results(self, results: List[dict]) -> str:
        """
        Format search results into a readable string.
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Formatted string containing search results
        """
        formatted_results = []

        # Loop through each search result and format it
        for result in results:
            formatted_result = f"""
                Title: {result.get('title', 'No title')}
                Link: {result.get('link', 'No link')}
                Snippet: {result.get('snippet', 'No snippet')}
                Date: {result.get('date', 'No date')}
                """
            formatted_results.append(formatted_result)
            
        return "\n---\n".join(formatted_results)
    
    # Define the generate_response method
    def generate_response(self, 
                         user_query: str,
                         search_results: Optional[List[dict]] = None,
                         model: str = "claude-3-opus-20240229",  # Updated to latest stable model or can change to other cheaper models like claude-3-5-sonnet-20241022
                         temperature: float = 0.7) -> str:
        """
        Generate a response using Claude, incorporating search results if available.
        
        Args:
            user_query: User's original query
            search_results: Optional list of search results
            model: Model to use for generation
            temperature: Temperature for generation
            
        Returns:
            Generated response string
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        if search_results:
            formatted_results = self.format_search_results(search_results)
            
            # System prompt with search results
            system_prompt = f"""You are an AI assistant with access to current information as of {current_date}.
                Below are relevant search results to help answer the user's query accurately.
                Use this information to provide an up-to-date and factual response.

                Search Results:
                {formatted_results}

                Please synthesize the search results and answer the user's query. Cite specific sources when appropriate."""
        else:
            system_prompt = f"You are an AI assistant. Today's date is {current_date}."
            
        try:
            # Generate response using Claude
            response = self.client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_query}
                ]
            )
            return response.content
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "Sorry, there was an error generating the response."
    
    # Define the process_query method
    def process_query(self, 
                     user_query: str,
                     search_enabled: bool = True,
                     num_results: int = 3) -> str:
        """
        Process a user query with optional web search augmentation.
        
        Args:
            user_query: User's query string
            search_enabled: Whether to perform web search
            num_results: Number of search results to use
            
        Returns:
            Generated response string
        """

        # Perform web search if enabled
        if search_enabled:
            try:
                search_results = self.search_web(user_query, num_results)
                return self.generate_response(user_query, search_results)
            except Exception as e:
                print(f"Search failed: {str(e)}")

                # Fallback to regular generation if search fails
                return self.generate_response(user_query)
        else:
            return self.generate_response(user_query)

# Example usage
if __name__ == "__main__":
    # Make sure you have set these environment variables
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    serper_key = os.getenv("SERPER_API_KEY")
    
    if not anthropic_key or not serper_key:
        print("Please set both ANTHROPIC_API_KEY and SERPER_API_KEY environment variables")
        exit(1)
    
    # Initialize the SearchAugmentedGeneration class
    sag = SearchAugmentedGeneration(
        anthropic_api_key=anthropic_key,
        serper_api_key=serper_key
    )
    
    try:
        # Process a query with search augmentation
        response = sag.process_query(
            "Jelaskan mengenai efisiensi kementrian lembaga yang dilakukan oleh presiden Prabowo Subianto", # Input search query here
            search_enabled=True, # Enable web search
            num_results=3 # Number of search results to use
        )
        print(response)
    except Exception as e:
        print(f"Error processing query: {str(e)}")

# # Example response
# """
# [TextBlock(citations=None, text='Berdasarkan hasil pencarian, Presiden Prabowo Subianto melakukan efisiensi anggaran di kementerian, lembaga, dan daerah dengan tujuan untuk menghemat anggaran negara. Beberapa poin penting terkait efisiensi yang dilakukan oleh Presiden Prabowo:\n\n1.
#  Target penghematan anggaran berbeda-beda untuk setiap kementerian dan lembaga. Contohnya, Kementerian Pekerjaan Umum akan terdampak efisiensi atau pemangkasan anggaran. (Sumber: Tempo.co, 5 Februari 2025)\n\n2.
#  Presiden Prabowo menyinggung adanya "raja kecil" yang melawan upaya efisiensi anggaran. Beliau menegaskan bahwa efisiensi dilakukan untuk kepentingan masyarakat. (Sumber: detik.com, 3 hari yang lalu)\n\n3.
#  Dalam pidatonya, Presiden Prabowo menekankan pentingnya efisiensi dalam penggunaan Anggaran Pendapatan dan Belanja Negara (APBN) kepada seluruh jajaran pemerintahan. (Sumber: bpkp.go.id, 11 Desember 2024)\n\n
#  Efisiensi anggaran yang dilakukan oleh Presiden Prabowo bertujuan untuk mengoptimalkan penggunaan dana APBN agar lebih tepat sasaran dan bermanfaat bagi masyarakat.
#  Langkah ini juga diambil untuk mengatasi pemborosan dan penyalahgunaan anggaran di berbagai instansi pemerintahan.', type='text')]
# """