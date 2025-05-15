import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.types import CrawlResult
from typing import List
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig


async def demo_basic_crawl():
    """Basic web crawling with markdown generation"""
    print("\n ===== 1. Basic web crawling =====")

    async with AsyncWebCrawler() as crawler:
        results: List[CrawlResult] = await crawler.arun(
            url="https://news.ycombinator.com/"
        )

        for i, result in enumerate(results):
            print(f"Result: {i + 1}:")
            print(f"Successfully crawled: {result.success}")

            if result.success:
                print(f"Markdown length: {len(result.markdown.raw_markdown)} characters")
                print(f"First 100 characters: {result.markdown.raw_markdown[:100]}")
            else:
                print("Failed to crawl the page")

async def demo_parallel_crawl():
    """Crawl multiple URLs in parallel"""
    print("\n ===== 2. Parallel web crawling =====")

    urls = [
        "https://news.ycombinator.com/",
        "https://example.com/",
        "https://httpbin.org/html"
    ]

    async with AsyncWebCrawler() as crawler:
        results: List[CrawlResult] = await crawler.arun_many(
            urls=urls
        )

        print(f"Crawled {len(results)} URLs in parallel")
        for i, result in enumerate(results):
            print(
                f" {i + 1}. {result.url} - {'Sucess' if result.success else 'Failed'}"
            )
            
            
            

async def main():
    """Run all demo functions sequentially"""
    print("=== Comprehensive Crawl4AI Demo ===")
    print("Note: Some examples require API keys or other configuration")

    # Run all demos
    # await demo_basic_crawl()
    await demo_parallel_crawl()


    # Clean up any temp files that may have been created
    print("\n ===== Demo Completed =====")
    print("Check for any generated files (screenshots, PDFs, etc.) in the current directory")

    
if __name__ == "__main__":
    asyncio.run(main())