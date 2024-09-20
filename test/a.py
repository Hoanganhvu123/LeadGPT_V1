import asyncio
from playwright.async_api import async_playwright

async def test_browser():
    async with async_playwright() as p:
        # Launch Chrome browser
        browser = await p.chromium.launch(channel="chrome", headless=False)
        # Create a new page
        page = await browser.new_page()
        # Navigate to Google
        await page.goto("https://www.google.com")
        
        try:
            # Wait for the search input to be visible and interactable
            await page.wait_for_selector('input[name="q"]', state="visible", timeout=60000)
            
            # Type a search query
            await page.fill('input[name="q"]', "Shopee")
            # Press Enter to search
            await page.press('input[name="q"]', "Enter")
            
            # Wait for the search results to load
            await page.wait_for_load_state("networkidle", timeout=60000)
            
            # Wait for search results to appear
            await page.wait_for_selector('#search', timeout=60000)
            
            # Retrieve and print the page title
            title = await page.title()
            print(f"Search results page title: {title}")
            
            # Retrieve and print the first search result
            first_result = await page.query_selector('.g')
            if first_result:
                result_text = await first_result.inner_text()
                print(f"First search result: {result_text}")
            else:
                print("No search results found")
            
            # Keep the browser open for visualization
            await asyncio.sleep(10)  # Wait for 10 seconds before closing
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            # Close the browser
            await browser.close()

if __name__ == "__main__":
    asyncio.run(test_browser())
