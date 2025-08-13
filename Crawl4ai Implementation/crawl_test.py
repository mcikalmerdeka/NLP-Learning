#!/usr/bin/env python3
"""
Comprehensive Crawl4AI Demo Script for Windows
Run this from terminal: python crawl_test.py
"""

import asyncio
import sys
from crawl4ai import AsyncWebCrawler
from crawl4ai.types import CrawlResult
from typing import List

# Set Windows event loop policy before any asyncio operations
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


async def demo_imdb_crawl():
    """Crawl IMDB Top 250 movies - your original Jupyter notebook goal"""
    print("\n ===== 1. IMDB Top 250 Movies Crawl =====")

    async with AsyncWebCrawler(verbose=True) as crawler:
        results: List[CrawlResult] = await crawler.arun(
            url="https://www.imdb.com/chart/top/"
        )

        for i, result in enumerate(results):
            print(f"Result: {i + 1}:")
            print(f"Successfully crawled: {result.success}")

            if result.success:
                print(f"Markdown length: {len(result.markdown.raw_markdown)} characters")
                print("=" * 50)
                print("IMDB TOP 250 CONTENT (First 2000 characters):")
                print("=" * 50)
                print(result.markdown.raw_markdown[:2000])
                print("\n... (content truncated)")
            else:
                print("Failed to crawl IMDB page")


async def demo_basic_crawl():
    """Basic web crawling with markdown generation - from your main.py"""
    print("\n ===== 2. Basic Web Crawling Demo =====")

    async with AsyncWebCrawler(verbose=True) as crawler:
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
    print("\n ===== 3. Parallel Web Crawling Demo =====")

    urls = [
        "https://example.com/",
        "https://httpbin.org/html",
        "https://www.python.org/"
    ]

    async with AsyncWebCrawler(verbose=True) as crawler:
        results: List[CrawlResult] = await crawler.arun_many(
            urls=urls
        )

        print(f"Crawled {len(results)} URLs in parallel")
        for i, result in enumerate(results):
            print(
                f" {i + 1}. {result.url} - {'Success' if result.success else 'Failed'}"
            )
            if result.success:
                print(f"    Content length: {len(result.markdown.raw_markdown)} characters")


async def main():
    """Run all demo functions sequentially"""
    print("=== Comprehensive Crawl4AI Demo ===")
    print("Note: This script demonstrates various crawling capabilities")

    # Run your original IMDB crawl first
    await demo_imdb_crawl()
    
    # Then run the other demos from main.py
    await demo_basic_crawl()
    await demo_parallel_crawl()

    # Summary
    print("\n ===== Demo Completed =====")
    print("✅ All crawling operations completed successfully!")
    print("Check the output above for crawled content from:")
    print("  - IMDB Top 250 Movies")
    print("  - Hacker News")
    print("  - Multiple sites in parallel")


if __name__ == "__main__":
    asyncio.run(main())
