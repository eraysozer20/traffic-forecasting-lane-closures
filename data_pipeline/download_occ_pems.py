import os
import time
from playwright.sync_api import sync_playwright

# --- CONFIG ---
PEMS_USERNAME = os.environ.get("PEMS_USERNAME", "")
PEMS_PASSWORD = os.environ.get("PEMS_PASSWORD", "")
USERNAME = PEMS_USERNAME
PASSWORD = PEMS_PASSWORD
DISTRICT_ID = 8
YEAR = 2024
DOWNLOAD_DIR = os.path.join(os.getcwd(), "pems_data_final")

if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

def run_pems():
    with sync_playwright() as p:
        # headless=False lets you watch the magic happen
        browser = p.chromium.launch(headless=False, slow_mo=500)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        # 1. Login
        print("Logging in...")
        page.goto("https://pems.dot.ca.gov/")
        page.locator('input[name="username"]').fill(USERNAME)
        page.locator('input[name="password"]').fill(PASSWORD)
        page.get_by_role("button", name="Login").click()
        page.wait_for_selector("text=Logout")
        print("✅ Login Successful")

        # 2. Go to District 8 Clearinghouse
        page.goto(f"https://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id={DISTRICT_ID}")
        
        # 3. Loop through months (0=Jan, 1=Feb... 5=Jun)
        for m_idx in range(0, 6):
            print(f"\n--- 📅 Triggering Month {m_idx + 1} ---")
            
            # Execute the exact JS command you found
            # Format: processFiles(district, time_id, year, type, format, month_index)
            js_call = f"processFiles('{DISTRICT_ID}', '', {YEAR}, 'station_5min', 'text', {m_idx});"
            page.evaluate(js_call)
            
            # Wait for the "Available Files" table to refresh
            time.sleep(5)

            # 4. Find and download all visible station_5min links
            links = page.locator("a[href*='download']").all()
            
            # Filter for correct month in filename (e.g., _2024_01_)
            month_str = f"_{YEAR}_{m_idx + 1:02d}_"
            valid_files = [l for l in links if month_str in l.inner_text() and ".txt.gz" in l.inner_text()]
            
            print(f"Found {len(valid_files)} files.")

            for link in valid_files:
                file_name = link.inner_text().strip()
                save_path = os.path.join(DOWNLOAD_DIR, file_name)

                if os.path.exists(save_path):
                    continue

                print(f"📥 Downloading: {file_name}")
                try:
                    with page.expect_download(timeout=60000) as download_info:
                        link.click()
                    download = download_info.value
                    download.save_as(save_path)
                    time.sleep(1) # Rate limit protection
                except Exception as e:
                    print(f"❌ Failed {file_name}: {e}")

        browser.close()

if __name__ == "__main__":
    run_pems()