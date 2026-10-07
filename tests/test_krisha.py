import unittest

from services.krisha import extract_image_urls, validate_listing_url


class ValidateListingUrlTests(unittest.TestCase):
    def test_accepts_listing(self) -> None:
        self.assertEqual(
            validate_listing_url("https://www.krisha.kz/a/show/12345678/?utm_source=test"),
            "https://krisha.kz/a/show/12345678",
        )

    def test_rejects_wrong_host_and_path(self) -> None:
        self.assertIsNone(validate_listing_url("https://evil.example/a/show/12345678"))
        self.assertIsNone(validate_listing_url("https://krisha.kz/prodazha/kvartiry/"))
        self.assertIsNone(validate_listing_url("javascript:alert(1)"))


class ExtractImagesTests(unittest.TestCase):
    def test_extracts_multiple_sources_and_deduplicates(self) -> None:
        html = """
        <html><head>
          <meta property="og:image" content="https://photos.krisha.kz/a/main.jpg?w=400">
          <script type="application/ld+json">
            {"@type":"Product","image":[
              "https://photos.krisha.kz/a/main.jpg?w=1200",
              "https://photos.krisha.kz/a/second.webp"
            ]}
          </script>
        </head><body>
          <img src="/assets/thumb.png" srcset="https://photos.krisha.kz/a/small.jpg 400w,
            https://photos.krisha.kz/a/large.jpg 1600w">
        </body></html>
        """
        urls = extract_image_urls(html, "https://krisha.kz/a/show/123")
        self.assertIn("https://photos.krisha.kz/a/main.jpg", urls)
        self.assertIn("https://photos.krisha.kz/a/second.webp", urls)
        self.assertIn("https://photos.krisha.kz/a/large.jpg", urls)
        self.assertEqual(urls.count("https://photos.krisha.kz/a/main.jpg"), 1)


if __name__ == "__main__":
    unittest.main()
