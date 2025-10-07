import gradio as gr
import requests

def add_protocol(url):
    """Ensure we return a proper absolute URL string (or an empty string if no URL)."""
    if isinstance(url, list):
        url = url[0] if url else ""
    if not isinstance(url, str):
        url = str(url)
    if not url.strip():
        return ""
    if not (url.startswith("http://") or url.startswith("https://")):
        return "https://" + url
    return url

def link_or_text(url, label):
    if url:
        return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>'
    else:
        return f"{label}: N/A"

def process_field(field):
    if isinstance(field, list):
        return ", ".join([str(x).strip() for x in field])
    return str(field).strip()

def search_companies(ln_url, num_results):
    api_url = "http://localhost:8000/api/search-by-linkedin/"
    headers = {"Content-Type": "application/json"}
    payload = {"ln_url": ln_url, "k": num_results}

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        data = response.json()
        results = data.get('results', [])
        if not results:
            return "No companies found!", "", []
        
        all_cards = ""
        industries_set = set()
        locations_set = set()
        sizes_set = set()
        
        for company in results:
            company_name = company.get('cmp_name', 'N/A')
            raw_industries = company.get('cmp_ind', [])
            
            # Process industries for display
            display_industry = process_field(raw_industries)
            
            # Process industries for filtering
            if isinstance(raw_industries, str):
                raw_industries = [raw_industries]
            elif not isinstance(raw_industries, list):
                raw_industries = []
                
            cleaned_industries = [str(x).strip() for x in raw_industries]
            for industry in cleaned_industries:
                industries_set.add(industry)
            
            desc = company.get('cmp_desc', '')
            display_location = company.get('cmp_loc_country_norm', 'N/A')
            filter_location = company.get('cmp_loc_country_norm', 'N/A')
            locations_set.add(filter_location)
            sizes_set.add(str(company.get('cmp_size', 'N/A')))
            
            fb_url = add_protocol(company.get('cmp_fb_urls', ''))
            #insta_url = add_protocol(company.get('cmp_insta_urls', ''))
            twitter_url = add_protocol(company.get('cmp_twitter_urls', ''))
            ln_url_company = add_protocol(company.get('cmp_ln_url', ''))
            website_url = add_protocol(company.get('cmp_web_urls', ''))
            
            logo_url = company.get('cmp_logo', '')
            if not logo_url or logo_url == "N/A":
                logo_url = "https://via.placeholder.com/100x100.png?text=Logo"
            
            card = f"""
<hr>
<img src="{logo_url}" width="100" height="100" alt="Logo">
<h3>🏢 {company_name}</h3>
<p><strong>Industry:</strong> {display_industry}</p>
<p><strong>Location:</strong> {display_location}</p>
<p><em>{desc}</em></p>
<p><strong>Social Media:</strong></p>
<ul style="list-style-type:none; padding: 0;">
  <li>{link_or_text(fb_url, "Facebook")}</li>
  <li>{link_or_text(twitter_url, "Twitter")}</li>
  <li>{link_or_text(ln_url_company, "LinkedIn")}</li>
  <li>{link_or_text(website_url, "Website")}</li>
</ul>
"""
            all_cards += card
        
        return f"Found {len(results)} matching companies.", all_cards, results

    except Exception as e:
        return f"Error: {str(e)}", "", []

def filter_results(results, industry_filter, location_filter, size_filter):
    filtered = []
    for company in results:
        # Process industries
        raw_industries = company.get('cmp_ind', [])
        if isinstance(raw_industries, str):
            raw_industries = [raw_industries]
        elif not isinstance(raw_industries, list):
            raw_industries = []
        company_industries = [str(x).strip() for x in raw_industries]
        
        location = company.get('cmp_loc_country_norm', 'N/A')
        size = str(company.get('cmp_size', 'N/A'))
        
        industry_match = (industry_filter == "All") or (industry_filter in company_industries)
        location_match = (location_filter == "All") or (location == location_filter)
        size_match = (size_filter == "All") or (size == size_filter)
        
        if industry_match and location_match and size_match:
            filtered.append(company)
    
    all_cards = ""
    for company in filtered:
        company_name = company.get('cmp_name', 'N/A')
        display_industry = process_field(company.get('cmp_ind', 'N/A'))
        desc = company.get('cmp_desc', '')
        display_location = company.get('cmp_loc', 'N/A')
        
        fb_url = add_protocol(company.get('cmp_fb_urls', ''))
        #insta_url = add_protocol(company.get('cmp_insta_urls', ''))
        twitter_url = add_protocol(company.get('cmp_twitter_urls', ''))
        ln_url_company = add_protocol(company.get('cmp_ln_url', ''))
        website_url = add_protocol(company.get('cmp_web_urls', ''))
        
        logo_url = company.get('cmp_logo', '')
        if not logo_url or logo_url == "N/A":
            logo_url = "https://via.placeholder.com/100x100.png?text=Logo"
        
        card = f"""
<hr>
<img src="{logo_url}" width="100" height="100" alt="Logo">
<h3>🏢 {company_name}</h3>
<p><strong>Industry:</strong> {display_industry}</p>
<p><strong>Location:</strong> {display_location}</p>
<p><em>{desc}</em></p>
<p><strong>Social Media:</strong></p>
<ul style="list-style-type:none; padding: 0;">
  <li>{link_or_text(fb_url, "Facebook")}</li>
  <li>{link_or_text(twitter_url, "Twitter")}</li>
  <li>{link_or_text(ln_url_company, "LinkedIn")}</li>
  <li>{link_or_text(website_url, "Website")}</li>
</ul>
"""
        all_cards += card
    return all_cards

def get_filter_options(results):
    industries = set()
    locations = set()
    sizes = set()
    
    for company in results:
        # Handle industries
        raw_industries = company.get('cmp_ind', [])
        if isinstance(raw_industries, str):
            raw_industries = [raw_industries]
        elif not isinstance(raw_industries, list):
            raw_industries = []
            
        for industry in raw_industries:
            industries.add(str(industry).strip())
        
        # Handle locations
        location = company.get('cmp_loc_country_norm', 'N/A')
        locations.add(location)
        
        # Handle sizes
        size = str(company.get('cmp_size', 'N/A'))
        sizes.add(size)
    
    return (
        ["All"] + sorted(industries),
        ["All"] + sorted(locations),
        ["All"] + sorted(sizes)
    )

def update_filter_options(results):
    industry_options, location_options, size_options = get_filter_options(results)
    return (
        gr.update(choices=industry_options, value="All"),
        gr.update(choices=location_options, value="All"),
        gr.update(choices=size_options, value="All")
    )

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Company's Similar Profile Search")
    gr.Markdown("Enter the LinkedIn URL of a company to find similar companies.")
    
    # Input Section
    with gr.Row():
        ln_input = gr.Textbox(label="🔗 LinkedIn URL", placeholder="https://linkedin.com/company/...")
        num_slider = gr.Slider(1, 500, value=10, label="Number of Recommendations")
    
    # Search Button
    search_btn = gr.Button("Search", variant="primary")
    
    # Horizontal Filter Bar
    with gr.Group(visible=False) as filter_group:
        with gr.Row(variant="panel"):
            with gr.Column(scale=3):
                industry_dropdown = gr.Dropdown(
                    ["All"],
                    label=" Industry",
                    info="Filter by industry sector",
                    interactive=True
                )
            with gr.Column(scale=3):
                location_dropdown = gr.Dropdown(
                    ["All"],
                    label=" Location",
                    info="Filter by country/region",
                    interactive=True
                )
            with gr.Column(scale=3):
                size_dropdown = gr.Dropdown(
                    ["All"],
                    label=" Company Size",
                    info="Filter by employee count",
                    interactive=True
                )
            with gr.Column(scale=1):
                filter_btn = gr.Button(
                    "Apply Filters",
                    variant="primary",
                    size="sm",
                    min_width=100
                )

    # Results Section
    status_output = gr.Textbox(label="Status", visible=True)
    cards_output = gr.HTML(label="Recommended Companies")
    results_state = gr.State([])
    def toggle_filters(results):
        return gr.update(visible=bool(results))

    # Event handling
    search_btn.click(
        search_companies,
        [ln_input, num_slider],
        [status_output, cards_output, results_state]
    ).then(
        update_filter_options,
        results_state,
        [industry_dropdown, location_dropdown, size_dropdown]
    ).then(
        toggle_filters,
        results_state,
        filter_group
    )

    filter_btn.click(
        filter_results,
        [results_state, industry_dropdown, location_dropdown, size_dropdown],
        cards_output
    )

demo.launch(server_name="0.0.0.0", server_port=7861)