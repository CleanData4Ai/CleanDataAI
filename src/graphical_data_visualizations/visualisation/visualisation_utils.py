# Import necessary modules and libraries for visualization utilities

from typing import List, Dict, Union
import plotly.graph_objs as go
import plotly.offline as pyo
import pandas as pd  # For data manipulation and analysis using dataframes.
import matplotlib.pyplot as plt  # For creating visualizations and plots in Python.
from bs4 import BeautifulSoup  # For parsing and manipulating HTML content.
import io  # For handling in-memory byte streams.
import base64  # For encoding binary data into base64 format, often used for embedding images in HTML.
import re
import seaborn as sns


"""

VisualisationUtils Class Overview:

The `VisualisationUtils` class provides essential utility functions for creating and managing visualizations 
and HTML content in a web-based application. This class supports features like:

- Generating HTML for sidebar sections and dashboards.
- Embedding Matplotlib figures and pandas DataFrames into HTML files.
- Adding custom dashboard cards and textual content to HTML sections.
- Converting Matplotlib figures into base64-encoded images for seamless embedding in HTML.
- Managing the structure, styling, and overall content of HTML files, with specific focus on sidebar configurations and visual elements.

The class leverages libraries such as BeautifulSoup for HTML manipulation, pandas for data handling, 
and Matplotlib for generating visualizations, making it highly effective for integrating dynamic content 
into web dashboards and reports.

"""


class VisualisationUtils:

    def __init__(self):

        """
        Initializes the VisualisationUtils class, setting up utility methods for managing visualizations and HTML content.
        No parameters are required for initialization.
        """

        pass

    def ncr_side_bar_sections(self, options: list[tuple[str, list[str]]]) -> str:


        """

        Generates the HTML content for sidebar sections.

        Args:
            options (list of tuples): A list where each tuple contains a main section and its subsections.

        Returns:
            str: A string containing the HTML code for all the sections.
        """
        section_template = '''
        
        <head>
            <!-- Link to Google Fonts for a better typography experience -->
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
        </head>

          
        <section id="{section_id}" class="p-4 mb-4 bg-white rounded-lg shadow-sm border border-gray-200">
              <div style="font-family: 'Inter', sans-serif;">{subsections_html}</div>
        </section>
        


        '''


        subsection_template = '''
            <div id="{subsection_id}" class="p-4 mb-4 bg-white rounded-lg shadow-sm border border-gray-200">
                
            </div>
        '''

        sections_html = []
        for section_title, subsections in options:
            #section_id = section_title.replace(' ', '').lower()
            section_id = section_title

            # Generate subsections HTML only if there are subsections
            subsections_html = ''
            if subsections:
                for subsection in subsections:
                    #subsection_id = subsection.replace(' ', '').lower()
                    subsections_html += subsection_template.format(subsection_id=subsection,
                                                                   subsection_title=subsection)

            # Format the section with its title and (if present) subsections
            section_html = section_template.format(section_id=section_id, section_title=section_title,
                                                   subsections_html=subsections_html)
            sections_html.append(section_html)

        return '\n'.join(sections_html)

    def ncr_slide_bar_menu(self, title: str, options: list[tuple[str, list[str]]], logo_url: str = None) -> tuple[str,str,str,str]:

        """
        Generates a complete HTML layout featuring a sidebar with menu options and content sections.

        Args:
            title (str): The title of the sidebar menu.
            options (list of tuples): A list of tuples where each tuple contains a main menu option and its sub-options.
            logo_url (str, optional): The URL of the logo image. Defaults to None.

        Returns:
            str: The full HTML code for the entire layout, including the sidebar and sections.
        """
        
        sidebar_html = '''


            <div id="nav-bar">
                <div id="nav-header" class="d-flex align-items-center">
                    {logo_section}
                    <h2 id="nav-title">{title}</h2>
                </div>
                <hr>
                <div id="nav-content">
                    <ul class="list-unstyled">
                        {options_list}
                    </ul>
                </div>
            </div>
            '''
            

        # Create the logo section if a logo URL is provided
        logo_section = f'''
                <a href="/" class="d-flex align-items-center mb-4">
                    <img src="{logo_url}" alt="Logo" class="img-fluid rounded-circle" style="max-height: 200px;">
                </a>
            ''' if logo_url else ''

        # Generate the list of options, including dropdown functionality
        options_list = ''
        for option, sub_options in options:
            if sub_options:
                # Main option with sub-options (dropdown)
                options_list += f'''
                    <li class="nav-item dropdown">
                        <a href="#" class="d-flex align-items-center nav-link dropdown-toggle nav-button" id="{option}-dropdown" role="button" data-bs-toggle="dropdown" aria-expanded="false">
                            <i class="fas nav-icon nav-button"></i><span>{option}</span>  
                        </a>
                        <ul class="dropdown-menu" aria-labelledby="{option}-dropdown">
                    '''
                for sub_option in sub_options:
                    # Link to the specific subsection within the main section
                    options_list += f'''
                            <li><a class="dropdown-item" href="#{sub_option}">{sub_option}</a></li>
                        '''
                options_list += '</ul></li>'
            else:
                # If no sub-options, show the main option with an anchor link to its corresponding section
                options_list += f'''
                    <li class="nav-item">
                        <a href="#{option}" class="d-flex align-items-center nav-link nav-button">
                            <i class="fas nav-icon nav-button"></i><span>{option}</span>  
                        </a>
                    </li>
                    '''

        # Generate the HTML for all sections using the side_bar_sections function
        sections_html = self.ncr_side_bar_sections(options)

        return logo_section, title, options_list, sections_html
    

    def insert_professional_text(self, html_code_string: str, section_id: str, text: str, 
                             heading_level: int = 3, add_separator: bool = True) -> str:
        
        """
        Inserts text into the specified section of an HTML code string with professional styling.
        
        This function adds text to the target section with professional typography, appropriate spacing,
        and optional visual separators to enhance readability and visual hierarchy.
        
        Parameters:
        - html_code_string (str): The HTML code as a string.
        - section_id (str): The ID of the section where the text will be inserted.
        - text (str): The text content to be inserted into the section.
        - heading_level (int): The heading level (1-6) if the text should be a heading (default is 3).
        - add_separator (bool): Whether to add a horizontal line separator after the text (default is True).
        
        Returns:
        - str: The updated HTML content with the professionally styled text inserted.
        """
        
        # Parse the HTML string using BeautifulSoup
        soup = BeautifulSoup(html_code_string, 'lxml')
        
        # Locate the section by its ID
        section = soup.find(id=section_id)
        
        if not section:
            print(f"Warning: Section with ID '{section_id}' not found.")
            return html_code_string
        
        # Convert the text to paragraphs if it contains multiple lines
        paragraphs = text.split('\n\n')
        
        # Create a container for the text content
        content_container = soup.new_tag('div', **{'class': 'professional-text-container my-4'})
        
        # Add each paragraph with proper styling
        for i, paragraph in enumerate(paragraphs):
            # First paragraph could be a heading if specified
            if i == 0 and heading_level >= 1 and heading_level <= 6:
                tag = soup.new_tag(f'h{heading_level}', **{
                    'class': 'text-xl font-semibold mb-3 text-gray-800'
                })
                tag.string = paragraph
            else:
                tag = soup.new_tag('p', **{
                    'class': 'text-base leading-relaxed mb-3 text-gray-700'
                })
                tag.string = paragraph
            
            content_container.append(tag)
        
        # Add a separator if requested
        if add_separator:
            separator = soup.new_tag('hr', **{
                'class': 'my-4 border-gray-200'
            })
            content_container.append(separator)
        
        # Append the content container to the section
        section.append(content_container)
        
        # Return the modified HTML as a string
        return str(soup)

    def fig_to_base64(self, fig: plt.Figure) -> str:

        """

        Converts a Matplotlib figure into a base64-encoded PNG image.

        Args:

            fig (plt.Figure): The Matplotlib figure to be converted.

        Returns:

            str: A base64-encoded string representing the PNG image of the figure.

        """
        # Create an in-memory bytes buffer to hold the image data
        img_bytes = io.BytesIO()

        # Save the figure to the buffer as a PNG
        fig.savefig(img_bytes, format='png')

        # Rewind the buffer to the beginning so the image data can be read
        img_bytes.seek(0)

        # Encode the image data as a base64 string and return it
        return base64.b64encode(img_bytes.read()).decode('utf-8')

    def ncr_plot(self, html_code_string: str, fig: Union['go.Figure', 'plt.Figure'], section_id: str,
                 description: str) -> str:
        """
        Inserts a plot (either Plotly or Matplotlib) into an HTML code string within a specified section.

        Args:
            html_code_string (str): The HTML code string where the plot will be inserted.
            fig (Union[go.Figure, plt.Figure]): The figure to insert, either as a Plotly or Matplotlib figure.
            section_id (str): The ID of the HTML section where the plot should be inserted.
            description (str): A description to display with the plot.

        Returns:
            str: Updated HTML code string with the plot added.
        """
        # Escape the section_id to avoid regex errors with special characters
        escaped_section_id = re.escape(section_id)

        # Search the HTML code for the section by ID
        section_pattern = re.compile(rf'(<([a-zA-Z]+)[^>]*\bid="{escaped_section_id}"[^>]*>)(.*?)(</\2>)', re.DOTALL)
        match = section_pattern.search(html_code_string)

        # If section not found, raise an error
        if match is None:
            raise ValueError(f"Section with ID '{section_id}' not found in HTML code string.")

        # Convert the figure to an HTML string
        if isinstance(fig, go.Figure):
            # Convert Plotly figure to HTML div
            plot_html = pyo.plot(fig, include_plotlyjs='cdn', output_type='div')
        elif isinstance(fig, plt.Figure):
            # Convert Matplotlib figure to base64 image tag
            plot_data = self.fig_to_base64(fig)
            img_tag = f'<img src="data:image/png;base64,{plot_data}" alt="Plot" class="img-fluid">'
            plot_html = img_tag
        elif isinstance(fig, str):
            # Use provided HTML string directly
            plot_html = fig
        else:
            raise ValueError(f"Unsupported plot type: {type(fig)}")

        # Apply animation classes for a smooth fade-in effect
        animation_class = 'animate__animated animate__fadeInUp'

        # Construct the HTML content for the plot and its description
        full_plot_html = f"""
            <div class="plot {animation_class}">
                <div class="description">{description}</div>
                <div class="plot-content">
                    {plot_html}
                </div>
            </div>
            """

        # Ensure all backslashes are escaped in the HTML content
        updated_section_html = f"{match.group(1)}{match.group(3)}{full_plot_html}{match.group(4)}"
        updated_section_html = re.sub(r"\\", r"\\\\", updated_section_html)

        # Replace the old section with the updated one in the HTML string
        updated_html_code = section_pattern.sub(updated_section_html, html_code_string)

        return updated_html_code



    def insert_plots_and_dataframes(self, html_string_code: str,
                                    sections: Dict[
                                        str, Dict[str, List[Union[go.Figure, plt.Figure, str, pd.DataFrame]]]],
                                    dataframe_name: str = 'dataframe') -> None:

        for section_id, content in sections.items():
            # Process descriptions
            descriptions = content.get('descriptions', [])
            for desc in descriptions:
                html_string_code = self.ncr_plot(html_string_code, desc, section_id, description="")

            # Process figures
            figures = content.get('figures', [])
            for figure in figures:
                html_string_code = self.ncr_plot(html_string_code, figure, section_id, description="")

            # Process dataframes
            dataframes = content.get('dataframes', [])
            for df in dataframes:
                html_string_code = self.ncr_dataframe(html_string_code, section_id, df, dataframe_name)

            return html_string_code

    def ncr_dataframe(self, html_code_string: str, section_id: str, dataframe: pd.DataFrame,
                      dataframe_name: str = 'dataframe') -> str:
        """
        Inserts a pandas DataFrame into the specified section of an HTML code string.

        Parameters:
        - html_code_string (str): The HTML code as a string.
        - section_id (str): The ID of the section where the DataFrame will be inserted.
        - dataframe (pd.DataFrame): The DataFrame to be inserted.
        - dataframe_name (str): Optional; a name for the DataFrame (default is 'dataframe').

        Returns:
        - str: The modified HTML code with the DataFrame inserted.
        """

        # Parse the HTML code string into a BeautifulSoup object
        soup = BeautifulSoup(html_code_string, 'lxml')

        # Locate the section where the DataFrame will be inserted
        section = soup.find(id=section_id)

        # If section not found, raise an error
        if section is None:
            raise ValueError(f"Section with ID '{section_id}' not found in HTML code string.")

        # Convert the DataFrame to HTML with custom styling classes
        df_html = dataframe.to_html(classes='custom-table table-responsive', index=False)

        # Create a div to contain the styled table
        new_content = soup.new_tag('div', **{'class': 'table-container'})
        new_content.append(BeautifulSoup(df_html, 'lxml'))

        # Append the new content to the section
        section.append(new_content)

        # Return the modified HTML as a string
        return str(soup)

    # this function will create our dashboard  interface
    def ncr_dashboard_dataframe(self, html_string, section_id, dataframe, dataframe_name='dataframe'):

        # Ensure the DataFrame has exactly one row
        if dataframe.shape[0] > 1:
            raise ValueError("DataFrame must contain exactly one row.")

        # Parse the HTML string
        soup = BeautifulSoup(html_string, 'lxml')

        # Locate the target section by its ID
        section = soup.find(id=section_id)

        if section:
            # Create a container for the dashboard card
            dashboard_card = soup.new_tag('div', **{'class': 'dashboard-card'})

            # Add each column's key-value pair to the dashboard card
            for col in dataframe.columns:
                value = dataframe.iloc[0][col]
                item_div = soup.new_tag('div', **{'class': 'dashboard-item'})
                key_div = soup.new_tag('div', **{'class': 'dashboard-key'})
                value_div = soup.new_tag('div', **{'class': 'dashboard-value'})

                key_div.string = col
                value_div.string = str(value)

                item_div.append(key_div)
                item_div.append(value_div)
                dashboard_card.append(item_div)

            # Append the dashboard card to the section
            section.append(dashboard_card)
        else:
            print(f"Section with ID '{section_id}' not found.")
            return None

        # Return the updated HTML content as a string
        return str(soup)

    def ncr_text(self, html_code_string: str, section_id: str, text: str, type: str = "h4") -> str:
        """
        Inserts text into the specified section of an HTML code string as a subheader.
        The subheader will always be placed at the top of the section, before any existing content.
        The text is styled professionally for better readability and visual appeal.

        Parameters:
        - html_code_string (str): The HTML code as a string.
        - section_id (str): The ID of the section where the text will be inserted.
        - text (str): The text to be inserted as a subheader.
        - type (str): The HTML tag type for the text (default is 'h4').

        Returns:
        - str: The modified HTML code with the new subheader inserted at the top of the section.
        """

        # Parse the HTML code string into a BeautifulSoup object
        soup = BeautifulSoup(html_code_string, 'lxml')

        # Locate the target section by its ID
        section = soup.find(id=section_id)

        if section:
            # Create the subheader HTML element with enhanced styling
            subheader_html = f'''
            <{type} class="subheader">
                <span class="subheader-text">{text}</span>
            </{type}>
            '''
            
            # Convert the HTML string into a BeautifulSoup Tag
            new_content = BeautifulSoup(subheader_html, 'lxml').contents[0]

            # Insert the new subheader at the top of the section
            section.insert(0, new_content)
        else:
            print(f"⚠️ Section with ID '{section_id}' not found.")
            return html_code_string  # Return the original HTML if section not found

        # Return the modified HTML as a string
        return str(soup)
    



    def ncr_text_3(self, html_code_string: str, section_id: str, text: str, type: str = "div") -> str:
        """
        Inserts text into the specified section of an HTML code string as properly formatted paragraphs.
        The text will always be placed at the bottom of the section, after any existing content.
        The text is styled professionally for better readability and visual appeal.
        
        Parameters:
        - html_code_string (str): The HTML code as a string.
        - section_id (str): The ID of the section where the text will be inserted.
        - text (str): The text to be inserted as formatted paragraphs.
        - type (str): The HTML container tag type for the text (default is 'div').
        
        Returns:
        - str: The modified HTML code with the new formatted text inserted at the bottom of the section.
        """
        
        # Parse the HTML code string into a BeautifulSoup object
        soup = BeautifulSoup(html_code_string, 'lxml')
        
        # Locate the target section by its ID
        section = soup.find(id=section_id)
        
        if section:
            # Process text to handle newlines and formatting
            formatted_text = text.replace("\t", "&emsp;")  # Convert tabs to spaces
            
            # Split text into paragraphs based on double newlines
            paragraphs = [p.strip() for p in formatted_text.split('\n\n') if p.strip()]
            
            # Create wrapper container
            wrapper = soup.new_tag(type, **{"class": "formatted-content"})
            wrapper['style'] = "font-size: 14px; line-height: 1.6; color: #333; margin-top: 20px;"

            # Process each paragraph and any lists within them
            for paragraph in paragraphs:
                if paragraph.strip().startswith(('- ', '• ', '* ', '1. ', '§')):  # Check for lists
                    lines = paragraph.split('\n')
                    list_tag = soup.new_tag('ul')
                    list_tag['style'] = "font-size: 14px; padding-left: 20px; margin: 10px 0;"

                    if any(line.strip().startswith(('1. ', '2. ')) for line in lines):  # Numbered list
                        list_tag = soup.new_tag('ol')

                    for line in lines:
                        line = line.strip()
                        if line:
                            if line.startswith(('- ', '• ', '* ')):
                                line = line[2:]
                            elif line.startswith('§'):
                                line = line[1:]
                            elif any(line.startswith(f"{i}. ") for i in range(1, 10)):
                                line = line[line.index('. ')+2:]
                                
                            item = soup.new_tag('li')
                            item.string = line
                            item['style'] = "margin-bottom: 5px; font-size: 14px;"
                            list_tag.append(item)

                    wrapper.append(list_tag)
                else:
                    # Normal paragraph handling
                    p_tag = soup.new_tag('p')
                    p_tag['style'] = "margin-bottom: 12px; font-size: 14px;"

                    # Correctly handling newlines by appending <br> tags
                    lines = paragraph.split('\n')
                    for i, line in enumerate(lines):
                        if i > 0:
                            p_tag.append(soup.new_tag("br"))  # Append <br> between lines
                        p_tag.append(line)
                    
                    wrapper.append(p_tag)
            
            # Insert the new formatted content at the bottom of the section
            section.append(wrapper)
            
        else:
            print(f"⚠️ Section with ID '{section_id}' not found.")
            return html_code_string  # Return the original HTML if section not found
        
        # Return the modified HTML as a string
        return str(soup)


    
    def ncr_text_2(self, html_code_string: str, section_id: str, text: str, tag_type: str = "div") -> str:
        """
        Inserts a styled text element into the specified section of an HTML code string.

        Parameters:
        - html_code_string (str): The HTML content as a string.
        - section_id (str): The ID of the section where the text will be inserted.
        - text (str): The text content to be inserted into the section.
        - tag_type (str): The HTML tag to wrap the text in (default is 'div').

        Returns:
        - str: The updated HTML content with the new text inserted at the correct section.
        """
        
        # Parse the HTML string using BeautifulSoup
        soup = BeautifulSoup(html_code_string, 'lxml')

        # Locate the section by its ID
        section = soup.find(id=section_id)
        
        if section:
            # Create a styled HTML element with the specified tag
            styled_html = f'<{tag_type} class="text-3xl font-semibold mb-4 text-dark">{text}</{tag_type}>'

            # Parse the new styled HTML into a BeautifulSoup Tag object
            new_content = BeautifulSoup(styled_html, 'lxml').contents[0]

            # Append the new content into the identified section
            section.append(new_content)
        else:
            # If the section is not found, log a message and return the original HTML
            print(f"Warning: Section with ID '{section_id}' not found.")
            return html_code_string

        # Return the modified HTML string
        return str(soup)

    def generate_html(self,
                      options: dict,  # Changed 'dictionary' to 'dict' for correct Python type hinting
                      base_template: str,  # Corrected type from 'string' to 'str'
                      css_code: str,  # Corrected type from 'string' to 'str'
                      title: str,
                      logo_url: str) -> None:  # Added return type hinting for clarity
        """
        Generates an HTML file with a sidebar, using a provided base template and custom CSS.

        This function dynamically creates an HTML page with a sidebar, which can be used
        to navigate through different sections. The layout and style of the page are customized
        based on a provided template and CSS code.

        Parameters:
        ----------
        options : dict
            A dictionary where the keys represent main menu options and the values are lists of sub-options.
            These options will be used to generate the sidebar navigation.

        base_template : str
            A string representing the base HTML template, containing placeholders such as
            '{{html_code}}' for the dynamic sidebar content and '{{css_code}}' for the embedded CSS styles.

        css_code : str
            A string containing custom CSS code that will be inserted into the final HTML file
            to style the page as required.

        output_path : str
            The file path where the generated HTML file will be saved. This should be a valid
            path where the application has write access.

        title : str, optional
            The title of the sidebar menu, displayed at the top of the navigation bar.
            Defaults to "Telefónica Arco Exploratory Data Analysis Framework".

        logo_url : str, optional
            The URL of the logo to be displayed at the top of the sidebar.
            Defaults to Telefónica's logo URL.

        Returns:
        -------
        None
            This function does not return any values. It creates and writes the HTML file
            to the specified location.
        """

        # Generate the HTML layout for the sidebar
        logo_section, title, options_list, sections_html = self.ncr_slide_bar_menu(title, options, logo_url)

        # Replace placeholders in the base template
        complete_html = base_template.replace("{{logo_section}}", logo_section) \
            .replace("{{title}}", title) \
            .replace("{{options_list}}", options_list) \
            .replace("{{sections_html}}", sections_html) \
            .replace("{{css_code}}", css_code)

        return complete_html
