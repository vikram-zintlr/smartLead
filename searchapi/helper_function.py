from urllib.parse import urlparse, urlunparse
from requests import get as request
from time import sleep
from random import randint
import re
from typing import Dict, List, Optional, Union, Tuple

linkedin_black_list = ["", "linkedin.com/profile/view", "linkedin.com/feed", "linkedin.com", "linkedin.com/groups", "linkedin.com/search/results","linkedin.com/sales/profile",
                       "linkedin.com/network-manager/company","linkedin.com/invite-connect/connections","linkedin.com/admin","linkedin.com/edit/forms","linkedin.com/details",
                       "linkedin.com/search/results", "linkedin.com/organization-guest/company","linkedin.com/edu/school", "linkedin.com/company", "linkedin.com/mwlite/company",
                       "linkedin.com/grp/home", "linkedin.com/authwall","linkedin.com/company/unavailable", "linkedin.com/sharing/share-offsite", "linkedin.com/sharearticle", 
                       "ca.linkedin.com/organization-guest/company","linkedin.com/m/company", "linkedin.com/grps", "linkedin.com/jobs/view", "linkedin.com/in/company", "linkedin.com/feed/update",
                       "linkedin.com/public-profile/in", "linkedin.com/uas/login", "linkedin.com//company", "linkedin.com/school", "linkedin.com/in/unavailable","linkedin.com/linkedin.com", 
                       "in.linkedin.com/company", "de.linkedin.com/organization-guest/company", "linkedin.com/mynetwork", "linkedin.com/today/author","linkedin.com/public-profile/settings",
                       "linkedin.com/hp", "linkedin.com/groupinvitation", "linkedin.com/cws/share", "au.linkedin.com/organization-guest/company","linkedin.com/home", "linkedin.com/nhome",
                       "linkedin.com/clinkedin.com/company", "ca.linkedin.com/company","linkedin.com/sales/profile","linkedin.com/sales/people","linkedin.com/profile/view","http://linkedin.com/sales", 
                       "linkedin.com//sales", "linkedin.com/pub/","linkedin.com/organization-guest/school", "no.linkedin.com/organization-guest/company"]


twitter_black_list = ["", "twitter.com/search", 'linkedin.com/company', "twitter.com/#!", 'twitter.com/personal','twitter.com/site', 'twitter.com/account','https://twitter.com',
                      'http://twitter.com',"twitter.com/facebook.com", "twitter.com/twitterapi","twitter.com/home","twitter.com/#","linkedin.com/in","twitter.com/intent",
                      "linkedin.com/in","twitter.com/https:/","twitter.com",'twitter.com/wix','twitter.com/wordpress','twitter.com/share','twitter.com/rockettheme',"twitter.com/premium_theme",
                      "twitter.com/wordpressdotcom","twitter.com/massive_studio","twitter.com/hover",'twitter.com/pleskbyodin','twitter.com/onlydomains',"twitter.com/site5","twitter.com/http:/",
                      'twitter.com/privacy', "twitter", "twitter.com","twitter.com/", "twitter.com/twitter.com", "twitter.com/hashtag", "twitter.com/linkedin.com", "twitter.com/twitter",
                      "twitter.com/shopify", "twitter.com/viadeo", "twitter.com/login", "twitter.com/theme_fusion"]


fb_black_list = ["", "web.facebook.com","facebook", "facebook.com/groups", "facebook.com/people","facebook.com/pg","facebook.com/profile.php","web.facebook.com/pg","facebook.com/group.php",
                 "facebook.com/pages","m.facebook.com/profile.php","linkedin.com/company", "facebook.com","m.facebook.com/pages","facebook.com/search","facebook.com/public","l.facebook.com/l.php", 
                 "facebook.com/settings", "web.facebook.com/groups","business.facebook.com/pg","facebook.com/", "web.facebook.com/pages", "m.facebook.com/pg", "fr-fr.facebook.com/pages", "facebook.com/facebook",
                 "facebook.com/dialog","en-gb.facebook.com/people","en-gb.facebook.com/pg","https://wfacebook.com",'http://facebook.com','nl-nl.facebook.com/pages','facebook.com/media','facebook.com/login.php',
                 'facebook.com/home.php#!','facebook.com/facebook.com','de-de.facebook.com/pages','facebook.com/business','facebook.com/messages','facebook.com/places','facebook.com/wix','facebook.com/google']


web_black_list_keywords = ["linkedin.com/company", "facebook.com", "linkedin.com", "twitter.com", "github.com"]

class URLProcessor:

    def __normalize_url(self, url: str) -> str:
        """
        Normalize the URL by removing schemes and 'www' and handling trailing slashes.
        Args:
            url (str): The URL to normalize.
        Returns:
            str: The normalized URL.
        """
        url = url.rstrip('/')
        url = re.sub(r'^https?://(www\.)?', '', url, flags=re.IGNORECASE)
        return url.lower()

    def __process_url(self, url: str, slash_count: int, add_http: bool, add_https: bool, allow_check: bool) -> str:
        """
        Process and normalize a URL.

        Args:
            url (str): The URL to process.
            slash_count (int): Number of slashes to retain in the URL.
            add_http (bool): Whether to add HTTP if missing.
            add_https (bool): Whether to add HTTPS if missing.
            allow_check (bool): Additional check for adding slashes or prefixes.

        Returns:
            str: Processed URL.
        """
        # Normalize and split the URL by '/'
        url = self.__normalize_url(url)
        url_parts = url.split('/')

        # Remove query parameters and unwanted characters from the last part of the URL
        if "?" in url_parts[-1]:
            url_parts[-1] = url_parts[-1].split("?")[0]

        # Adjust slash_count if 'http://' or 'https://' is present
        if ('https://' in url or 'http://' in url) and slash_count > 0:
            slash_count += 1

        # If allow_check is enabled, increase slash_count
        if allow_check:
            slash_count += 2

        # Reconstruct the URL using the specified number of slashes
        url_new = "/".join(url_parts[:min(slash_count + 1, len(url_parts))])

        # Add the correct protocol (http/https) if missing
        if "https://" not in url_new and "http://" not in url_new:
            prefix = "http://www." if (add_http or allow_check) else "https://www." if add_https else ""
            if "www." not in url_new:
                url_new = prefix + url_new
            else:
                url_new = ("http://" if add_http or allow_check else "https://") + url_new

        return url_new.lower(), slash_count


    def __check_url_exists(self,
                         url: str,
                         allow_sleep: bool = False,ssl_verify = True) -> Tuple[bool, Optional[str]]:
        """
        Check if a URL exists by sending a GET request.

        Args:
            url (str): The URL to check.
            allow_sleep (bool): Whether to allow sleep between requests.

        Returns:
            Tuple[bool, Optional[str]]: Tuple indicating whether the URL exists and any error message.
        """
        if not url:
            return False, "URL is None or empty"

        if allow_sleep:
            sleep(randint(1, 3))
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
        } 
        try:
            response = request("GET", url, headers=headers, verify=ssl_verify, allow_redirects=True)
            if response.status_code < 400:
                return True, None
        except Exception as e:
            return False, str(e)
        return False, f"URL not found: {url}"


    def __clean_url_input(
        self,
        url: Optional[str],
        slash_count: int = 1,
        allow_check: bool = False,
        add_http: bool = False,
        add_https: bool = False,
        keep_www: bool = False,
        key=None,
        black_list=[]) -> Tuple[Optional[str], Optional[str]]:
        """
        Clean and normalize a URL.

        Args:
            url (Optional[str]): The URL to clean.
            slash_count (int): Number of slashes to retain in the URL.
            allow_check (bool): Whether to check if the URL exists.
            add_http (bool): Whether to add HTTP if missing.
            add_https (bool): Whether to add HTTPS if missing.
            keep_www (bool): Whether to keep the 'www.' prefix.

        Returns:
            Tuple[Optional[str], Optional[str]]: Cleaned URL and any error message.
        """
        if url is None or not url:
            return None, "URL is None or empty"
        
        if black_list and url in black_list:
            return None, "URL is in the blacklist"

        # Check if the URL contains the required key (if provided)
        if key and key not in url.lower():
            return None, f"URL does not contain the required key: {key}"
        url_new, slash_count = self.__process_url(url, slash_count, add_http, add_https, allow_check)
        if allow_check:
            check, err = self.__check_url_exists(url_new)
            if check:
                return url_new, None
            if 'https://' not in url_new:
                url_new = url_new.replace('http', 'https')
                check1, err1 = self.__check_url_exists(url_new)
                if check1:
                    return url_new, None
                print(url_new)
                return url_new, f"URL error: {err1}, original URL = {url}"
            return url_new, f"URL error: {err}, original URL = {url}"
        return url_new, None

    def clean_url(self, url_data, slash_count: int = 1,allow_check=False,add_https=False,add_http=False, black_list=None, key=None):
        """
        Clean a single URL, a list of URLs, or a dictionary of URL lists.

        Args:
            data (Union[List[str], Dict[str, List[str]], str]): URLs to clean. It can be a string (single URL), a list of URLs, or a dictionary where values are lists of URLs.
            slash_count (int): Number of slashes to retain in each URL.
            black_list (List[str], optional): List of URLs to skip cleaning. Defaults to None.
            key (Optional[str], optional): A keyword that should be present in each URL for cleaning. Defaults to None.

        Returns:
            Optional[Union[List[str], Dict[str, List[str]], str]]: Cleaned URLs, either as a single cleaned URL, a cleaned list of URLs, or a dictionary with cleaned URLs.
        """

        # If data is a single URL (string)
        if isinstance(url_data, str):
            cleaned_url, err = self.__clean_url_input(url=url_data, slash_count=slash_count, add_http=add_http, add_https=add_https, black_list=black_list, key=key)
            return cleaned_url

        # If data is a list of URLs
        elif isinstance(url_data, list):
            cleaned_urls = []  
            for raw_url in url_data:
                # Attempt to clean the URL
                cleaned_url, err = self.__clean_url_input(url=raw_url, slash_count=slash_count,black_list=black_list,key=key)
                
                # If no error, add to cleaned_urls
                if err is None:
                    cleaned_urls.append(cleaned_url)

            return cleaned_urls

        # If data is a dictionary of URL lists
        elif isinstance(url_data, dict):
            for key in ["cmp_ln_url", "cmp_twitter_urls", "cmp_fb_urls", "cmp_web_urls_norm"]:
                if key in url_data and url_data[key]:
                    if key == "cmp_web_urls_norm":
                        url_data[key] = [self.__clean_url_input(u, 0, web_black_list_keywords)[0] for u in url_data[key]]
                    else:
                        # Clean each URL for social media URLs (or other URLs)
                        url_data[key] = [self.__clean_url_input(u, 1 if key in ["cmp_twitter_urls", "cmp_fb_urls"] else 2)[0] for u in url_data[key]]

            return url_data
        return None
    
    @staticmethod
    def normalize_linkedin_url(linkedin_url: Optional[str]) -> Optional[str]:
        """
        Normalize a LinkedIn URL by converting it to a standard format.

        Args:
            linkedin_url (Optional[str]): The LinkedIn URL to normalize.

        Returns:
            Optional[str]: Normalized LinkedIn URL.
        """
        if linkedin_url is None:
            return None

        parsed_url = urlparse(linkedin_url)
        normalized_netloc = 'linkedin.com'
        normalized_path = parsed_url.path.rstrip('/')
        normalized_url = urlunparse(('https', normalized_netloc, normalized_path, '', '', ''))

        return normalized_url

    @staticmethod
    def extract_linkedin_url(linkedin_url: Optional[str]) -> Optional[str]:
        """
        Extract and normalize a LinkedIn URL by removing unwanted prefixes and components.

        Args:
            linkedin_url (Optional[str]): The LinkedIn URL to extract.

        Returns:
            Optional[str]: Extracted and normalized LinkedIn URL.
        """

        # Remove any leading or trailing whitespace
        if linkedin_url:
            linkedin_url = linkedin_url.strip()
        else:
            return None

        # Remove any "http://" or "https://" from the beginning of the URL
        if linkedin_url.startswith("http://"):
            linkedin_url = linkedin_url[7:]
        elif linkedin_url.startswith("https://"):
            linkedin_url = linkedin_url[8:]

        # Remove any "www." from the beginning of the URL
        if linkedin_url.startswith("www."):
            linkedin_url = linkedin_url[4:]

        # Extract and retain only the necessary parts of the URL
        if "linkedin.com/" in linkedin_url:
            parts = linkedin_url.split("linkedin.com/", 1)
            linkedin_url = "linkedin.com/" + parts[1]
        elif "linkedin.com" in linkedin_url:
            parts = linkedin_url.split("linkedin.com", 1)
            linkedin_url = "linkedin.com" + parts[1]

        # Remove any query parameters from the end of the URL
        linkedin_url = linkedin_url.split("?", 1)[0]

        # Remove trailing slash if it exists
        if linkedin_url.endswith("/"):
            linkedin_url = linkedin_url[:-1]

        # Handle different subdomains and paths
        if linkedin_url.startswith("linkedin.com/"):
            linkedin_url = linkedin_url.split("/", 2)
            linkedin_url = "/".join(linkedin_url[:3])

        # Check the blacklist
        new_ln_url = URLProcessor().clean_url(url_data=linkedin_url, slash_count=2, black_list=linkedin_black_list, key="linkedin.com")
        if new_ln_url in linkedin_black_list:
            return None

        return new_ln_url

    @staticmethod
    def get_all_combinations_of_linkedin_urls(inp_url: Union[str, List[str]]) -> Optional[List[str]]:
        """
        Generate all possible combinations of LinkedIn URLs based on the input URL.

        Args:
            inp_url (Union[str, List[str]]): The input URL or a list of URLs.

        Returns:
            Optional[List[str]]: A list of URL combinations, or None if the input is invalid.
        """
        if not inp_url:
            return None

        if isinstance(inp_url, list) and inp_url:
            inp_url = inp_url[0]

        url= URLProcessor().clean_url(inp_url, slash_count=2)
        if url is None:
            return None

        return [
            url,
            f"http://{url}",
            f"https://{url}",
            f"http://www.{url}",
            f"https://www.{url}",
            f"www.{url}",
            f"{url}/",
            f"http://{url}/",
            f"https://{url}/",
            f"http://www.{url}/",
            f"https://www.{url}/",
            f"www.{url}/"
        ]

    
