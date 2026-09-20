"""Generate a small 5-page digital PDF for parse-stream E2E checks."""
import fitz

doc = fitz.open()
for i in range(1, 6):
    page = doc.new_page()
    page.insert_text((72, 72), f"Parse stream test — page {i} of 5", fontsize=18)
    page.insert_text((72, 130), f"Paragraph on page {i}. " * 20, fontsize=11)
doc.save("_e2e_5p.pdf")
print("wrote _e2e_5p.pdf")
