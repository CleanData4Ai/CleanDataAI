"""
Custom CSS Styling for Report Interface

This CSS code is designed to style the report interface, including the sidebar, content sections, and various UI elements. It incorporates modern design principles and animations to enhance the user experience. The stylesheet includes styles for layout, responsiveness, and visual effects.

CSS Components:

1. **Sidebar Container (#nav-bar)**:
   - **Background**: Applies a gradient background with a glowing effect.
   - **Dimensions**: Fixed width and height for a full-height sidebar.
   - **Transitions**: Smooth transitions for background and width changes.
   - **Box Shadow**: Multiple shadows for a 3D effect and a glowing effect.
   - **Overflow Handling**: Prevents horizontal overflow.

2. **Glowing Animation (@keyframes glow)**:
   - **Keyframes**: Defines a glowing effect with an animation that intensifies and then softens.
   - **Application**: Applied to the sidebar and logo for a dynamic visual effect.

3. **Logo and Title in Sidebar (#nav-header)**:
   - **Layout**: Centered content with increased logo size and shadow effects.
   - **Hover Effects**: Scales the logo and changes border on hover for interactive feedback.

4. **Sidebar Menu Options (#nav-content ul)**:
   - **List Styling**: Removes default list styles and aligns items to the left.
   - **Navigation Button Styles**: Styled for readability, interaction, and visual appeal.

5. **Radio Button and Dropdown Menu Styles**:
   - **Hidden Radio Buttons**: Custom label styles for radio button interaction.
   - **Dropdown Menu**: Ensures proper text wrapping and overflow handling.

6. **Main Section Content (section)**:
   - **Layout**: Adjusted margin and padding to accommodate the sidebar.
   - **Background and Shadow**: Light background and subtle shadow for content sections.

7. **Responsive Adjustments (@media queries)**:
   - **Sidebar**: Adjusts width and position for smaller screens.
   - **Toggle Button**: Displays a toggle button for sidebar visibility on small screens.

8. **Table Styles**:
   - **General Table Styling**: Collapsed borders, padding, and alignment for readability.
   - **Striped and Bordered Tables**: Alternating row colors and border styles.
   - **Responsive Tables**: Scrollable tables with custom scrollbar styling.

9. **Custom Table and Image Styling**:
   - **Custom Tables**: Light blue background and dark blue text for emphasis.
   - **Responsive Images**: Scaling and shadow effects for a polished look.

10. **Dashboard Cards**:
    - **Design**: Gradient background, rounded corners, and shadow effects.
    - **Hover Effect**: Lifts the card and enhances the shadow on hover for interactive feedback.

11. **Dashboard Key-Value Pair Styling**:
    - **Text Styling**: Differentiates key and value text with color and size.

12. **Primary Button Styling (.btn-primary)**:
    - **Design**: Bright blue background with rounded corners and hover effects.
    - **Hover Effect**: Darkens background and slightly scales the button on hover.

13. **Animations**:
    - **Fade-In Up**: Smooth entry animation for elements.

Note: Ensure that custom CSS code is inserted in the placeholder `{{css_code}}` without additional spaces or characters to maintain proper styling.
"""


css_code = """
    /* Sidebar Container */
    #nav-bar {
        background: linear-gradient(145deg, #007acc 0%, #e6f7ff 100%);
        width: 260px;
        height: 100vh;
        position: fixed;
        top: 0;
        left: 0;
        padding-top: 30px;
        transition: background 0.3s ease, width 0.3s ease;
        overflow-x: hidden;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3), 
                    inset 0 4px 10px rgba(0, 0, 0, 0.3),
                    0 0 15px rgba(0, 0, 0, 0.2),
                    0 0 20px rgba(0, 122, 255, 0.6); /* Added glowing effect */
        z-index: 1000;
    }

    /* Glowing Animation */
    @keyframes glow {
        0% {
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3), 
                        inset 0 4px 10px rgba(0, 0, 0, 0.3),
                        0 0 15px rgba(0, 0, 0, 0.2),
                        0 0 20px rgba(0, 122, 255, 0.6); /* Glowing effect */
        }
        50% {
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3), 
                        inset 0 4px 10px rgba(0, 0, 0, 0.3),
                        0 0 15px rgba(0, 0, 0, 0.2),
                        0 0 30px rgba(0, 122, 255, 0.8); /* Intense glowing effect */
        }
        100% {
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3), 
                        inset 0 4px 10px rgba(0, 0, 0, 0.3),
                        0 0 15px rgba(0, 0, 0, 0.2),
                        0 0 20px rgba(0, 122, 255, 0.6); /* Glowing effect */
        }
    }

    /* Apply glowing animation to the sidebar */
    #nav-bar {
        animation: glow 1.5s infinite; /* Apply glowing animation */
    }

    /* Logo and Title in Sidebar */
    #nav-header {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 20px;
        margin-bottom: 40px;
        text-align: center;
    }

    #nav-header img {
        max-height: 150px; /* Increase logo size */
        border-radius: 50%;
        border: 5px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.5), 
                    0 0 20px rgba(255, 255, 255, 0.8);
        animation: glow 1.5s infinite; /* Apply glowing animation */
    }

    #nav-header img:hover {
        transform: scale(1.1); /* Zoom effect on hover */
        border: 3px solid #80cfff; /* Border change on hover */
    }

    /* Glowing Animation */
    @keyframes glow {
        0% {
            box-shadow: 0 0 5px rgba(255, 255, 255, 0.8), 0 0 10px rgba(255, 255, 255, 0.6);
        }
        50% {
            box-shadow: 0 0 15px rgba(255, 255, 255, 1), 0 0 30px rgba(255, 255, 255, 0.8);
        }
        100% {
            box-shadow: 0 0 5px rgba(255, 255, 255, 0.8), 0 0 10px rgba(255, 255, 255, 0.6);
        }
    }

    /* Sidebar Menu Options */
    #nav-content ul {
        list-style-type: none;
        padding-left: 0; /* Adjust padding to create space on the left */
        margin-top: 20px;
        text-align: left; /* Align items to the left */
    }

    .nav-button {
        display: block;
        margin: 10px 0;
        padding: 12px 20px;
        text-align: left; /* Align button text to the left */
        text-decoration: none;
        color: #ffffff;
        position: relative;
        cursor: pointer;
        white-space: normal; /* Allow text to wrap */
        overflow-wrap: break-word; /* Break long words to avoid overflow */
    }

    /* Ensure dropdown items wrap correctly */
    .dropdown-menu {
        white-space: nowrap; /* Prevent line breaks within dropdown items */
    }

    .dropdown-item {
        white-space: normal; /* Allow text to wrap */
        overflow-wrap: break-word; /* Break long words to avoid overflow */
    }

    /* Radio Button Styles */
    .nav-item {
        position: relative;
    }

    .nav-item input[type="radio"] {
        position: absolute;
        opacity: 0;
        pointer-events: none;
    }

    .nav-item label {
        display: block;
        padding: 12px 20px;
        background: #007bff;
        color: #fff;
        border-radius: 4px;
        cursor: pointer;
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.2);
        transition: box-shadow 0.3s ease-in-out;
        white-space: normal; /* Allow text to wrap */
        overflow-wrap: break-word; /* Break long words to avoid overflow */
    }

    .nav-item input[type="radio"]:checked + label {
        box-shadow: 0 0 15px rgba(0, 123, 255, 0.8);
    }

    .nav-item .sub-options {
        display: none;
        padding-left: 20px;
    }

    .nav-item input[type="radio"]:checked ~ .sub-options {
        display: block;
    }

    .sub-options label {
        background: #e6f7ff;
        color: #007acc;
        padding: 10px;
        white-space: normal; /* Allow text to wrap */
        overflow-wrap: break-word; /* Break long words to avoid overflow */
    }

    .sub-options label:hover {
        background: rgba(255, 255, 255, 0.1);
    }

    /* Main Section Content */
    section {
        margin-left: 260px;
        padding: 30px;
        background-color: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        #nav-bar {
            width: 100%;
            height: auto;
            position: relative;
            box-shadow: none;
        }
        
        #nav-content {
            padding: 10px;
        }
        
        section {
            margin-left: 0;
        }
    }

    /* Sidebar Title Styling */
    
    #nav-title {
        font-size: 1rem; /* Larger font size for prominence */
        font-weight: 900; /* Extra bold text for impact */
        color: #ffffff; /* White color for contrast */
        text-transform: uppercase; /* Uppercase text for a formal appearance */
        letter-spacing: 1.5px; /* Increased letter spacing for clarity */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); /* Subtle shadow for depth */
        margin-bottom: 20px; /* Spacing below the title */
        animation: fadeIn 1s ease-out; /* Fade-in animation for smooth appearance */
    }

    /* Fade-In Animation */
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }

    /* Table Styles */
    .table {
        border-collapse: collapse;
        width: 100%;
        overflow-x: auto;
        margin: 20px 0;
        border: 1px solid #e0e0e0; /* Light grey border */
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }

    .table th, .table td {
        border: 1px solid #e0e0e0;
        padding: 12px;
        text-align: center;
        vertical-align: middle;
        font-size: 16px;
        font-weight: 500;
        color: #333;
    }

    .table th {
        background-color: #007bff; /* Blue header background */
        color: #ffffff; /* White text for contrast */
        border-bottom: 2px solid #0056b3;
    }

    .table td:hover {
        background-color: #f1f1f1; /* Light highlight on hover */
        cursor: pointer;
    }

    .table-striped tbody tr:nth-child(even) {
        background-color: #f9f9f9; /* Alternating row colors */
    }

    .table-bordered {
        border: 1px solid #e0e0e0;
    }

    .table-bordered th, .table-bordered td {
        border: 1px solid #e0e0e0;
    }

    .table-responsive {
        overflow-x: auto;
        max-width: 100%;
        margin: 20px auto;
    }

    .table-responsive::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }

    .table-responsive::-webkit-scrollbar-thumb {
        background-color: #cccccc; /* Custom scrollbar styling */
        border-radius: 10px;
    }

    /* Toggle Button for Small Screens */
    .nav-toggle-label {
        cursor: pointer;
        display: none; /* Hidden by default */
    }

    @media (max-width: 768px) {
        #nav-bar {
            width: 220px; /* Narrower sidebar on small screens */
        }

        section {
            margin-left: 240px; /* Adjusted margin */
        }

        .nav-toggle-label {
            display: block; /* Show toggle button */
        }

        #nav-toggle:checked ~ #nav-bar {
            width: 0; /* Collapse sidebar */
        }

        #nav-toggle:checked ~ section {
            margin-left: 0; /* Full width content */
        }
    }

    /* Custom Table Styling */
    .custom-table {
        border-collapse: collapse;
        width: 100%;
        margin-top: 20px;
        background-color: #f0faff; /* Light blue background */
        color: #003366; /* Dark blue text */
    }

    .custom-table th, .custom-table td {
        padding: 12px;
        border: 1px solid #d0e4f1; /* Light blue border */
        text-align: center;
        vertical-align: middle;
    }

    .custom-table th {
        background-color: #e6f4ff; /* Slightly darker header background */
        color: #003366; /* Dark blue header text */
        font-weight: bold;
    }

    .custom-table tr:nth-child(even) {
        background-color: #ffffff; /* White for even rows */
    }

    .table-container {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #d0e4f1;
        border-radius: 12px; /* Rounded edges for container */
    }

    /* Custom Styling for Plotted Images */
    .img-fluid {
        max-width: 100%;
        height: auto;
        border: 2px solid #d0e4f1; /* Light blue border */
        border-radius: 12px; /* Rounded corners */
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15); /* Enhanced shadow */
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        background-color: #f0faff; /* Light blue background behind image */
        padding: 5px; /* Padding around the image */
    }

    /* Hover Effect for Images */
    .img-fluid:hover {
        transform: scale(1.05); /* Slight zoom effect */
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.25); /* Enhanced shadow on hover */
        border-color: #80cfff; /* Slightly darker blue border on hover */
        background-color: #e6f4ff; /* Slightly darker blue background on hover */
    }

    /* General Styles for Dashboard Cards */
    .dashboard-card {
        background: linear-gradient(135deg, #ffffff 0%, #e0f7fa 100%); /* Gradient background */
        border-radius: 15px; /* Rounded edges */
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2); /* Deep shadow */
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        overflow: hidden;
        position: relative;
        padding: 20px;
        margin: 20px 0;
        animation: fadeInUp 0.6s ease-out; /* Smooth entry animation */
    }

    /* Hover Effect for Dashboard Cards */
    .dashboard-card:hover {
        transform: translateY(-10px); /* Lift effect on hover */
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3); /* Enhanced shadow on hover */
    }

    /* Key-Value Pair Styles in Dashboard Cards */
    .dashboard-item {
        margin-bottom: 15px;
        border-bottom: 1px solid #e0f7fa; /* Light cyan divider */
        padding-bottom: 10px;
    }

    .dashboard-key {
        font-size: 1.2rem;
        color: #003366; /* Dark blue for key text */
        font-weight: bold;
        margin-bottom: 5px;
    }

    .dashboard-value {
        font-size: 1.3rem;
        color: #333; /* Dark gray for value text */
        font-weight: normal;
    }

    /* Primary Button Styling */
    .btn-primary {
        background-color: #007bff; /* Bright blue */
        border-color: #007bff;
        font-size: 1rem;
        padding: 12px 24px;
        border-radius: 10px; /* Rounded corners */
        transition: background-color 0.3s ease, transform 0.3s ease;
    }

    .btn-primary:hover {
        background-color: #0056b3; /* Darker blue on hover */
        border-color: #004494;
        transform: scale(1.05); /* Slight zoom effect */
    }

    /* Animation Keyframes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
"""

css_code_with_black_side_bar = """
 /* Sidebar Container */
#nav-bar {
    background: linear-gradient(145deg, #1e3c72 0%, #2a5298 100%); /* Gradient with deeper blue tones */
    width: 400px; /* Slightly narrower for a modern look */
    height: 100vh;
    position: fixed;
    top: 0;
    left: 0;
    padding-top: 30px;
    transition: background 0.3s ease, width 0.3s ease;
    overflow-x: hidden;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3), 
                inset 0 4px 10px rgba(0, 0, 0, 0.3),
                0 0 15px rgba(0, 0, 0, 0.2),
                0 0 20px rgba(0, 122, 255, 0.6); /* Glowing effect */
    z-index: 1000;
}

/* Logo and Title in Sidebar */
#nav-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
    margin-bottom: 20px;
    text-align: center;
}

#nav-header img {
    max-height: 120px; /* Adjusted logo size */
    border-radius: 50%; /* Circular logo */
    border: 3px solid rgba(255, 255, 255, 0.5);
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3), 
                0 0 10px rgba(255, 255, 255, 0.5);
    transition: transform 0.3s ease, border-color 0.3s ease;
}

#nav-header img:hover {
    transform: scale(1.1); /* Zoom effect on hover */
    border-color: #80cfff; /* Border change on hover */
}

/* Sidebar Title Styling */
#nav-title {
    font-size: 1.5rem; /* Larger font size */
    font-weight: 900; /* Extra bold */
    color: #ffffff; /* White text */
    text-transform: uppercase; /* Uppercase text */
    letter-spacing: 2px; /* Increased letter spacing */
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); /* Subtle shadow */
    margin-top: 15px;
    animation: fadeIn 1s ease-out; /* Fade-in animation */
}

/* Sidebar Menu Options */
#nav-content ul {
    list-style-type: none;
    padding-left: 0;
    margin-top: 20px;
    text-align: left;
}

.nav-button {
    display: block;
    margin: 10px 0;
    padding: 12px 20px;
    text-align: left;
    text-decoration: none;
    color: #ffffff;
    position: relative;
    cursor: pointer;
    white-space: normal;
    overflow-wrap: break-word;
    transition: background-color 0.3s ease, color 0.3s ease;
}

.nav-button:hover {
    background-color: rgba(255, 255, 255, 0.1); /* Light hover effect */
    color: #80cfff; /* Highlight color */
}

/* Radio Button Styles */
.nav-item {
    position: relative;
}

.nav-item input[type="radio"] {
    position: absolute;
    opacity: 0;
    pointer-events: none;
}

.nav-item label {
    display: block;
    padding: 12px 20px;
    background: #007bff;
    color: #fff;
    border-radius: 4px;
    cursor: pointer;
    box-shadow: 0 0 5px rgba(0, 0, 0, 0.2);
    transition: box-shadow 0.3s ease-in-out, background-color 0.3s ease;
}

.nav-item input[type="radio"]:checked + label {
    box-shadow: 0 0 15px rgba(0, 123, 255, 0.8);
    background-color: #0056b3; /* Darker blue for selected state */
}

.nav-item .sub-options {
    display: none;
    padding-left: 20px;
}

.nav-item input[type="radio"]:checked ~ .sub-options {
    display: block;
}

.sub-options label {
    background: #e6f7ff;
    color: #007acc;
    padding: 10px;
    white-space: normal;
    overflow-wrap: break-word;
    transition: background-color 0.3s ease;
}

.sub-options label:hover {
    background: rgba(255, 255, 255, 0.1);
}

/* Main Section Content */
section {
    margin-left: 400px; /* Adjusted to match sidebar width */
    padding: 30px;
    background-color: #f8f9fa;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
}

/* Table Styles */
.table {
    border-collapse: collapse;
    width: 100%;
    overflow-x: auto;
    margin: 20px 0;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.table th, .table td {
    border: 1px solid #e0e0e0;
    padding: 12px;
    text-align: center;
    vertical-align: middle;
    font-size: 16px;
    font-weight: 500;
    color: #333;
}

.table th {
    background-color: #007bff;
    color: #ffffff;
    border-bottom: 2px solid #0056b3;
}

.table td:hover {
    background-color: #f1f1f1;
    cursor: pointer;
}

.table-striped tbody tr:nth-child(even) {
    background-color: #f9f9f9;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    #nav-bar {
        width: 100%;
        height: auto;
        position: relative;
        box-shadow: none;
    }

    section {
        margin-left: 0;
    }

    .nav-toggle-label {
        display: block;
    }

    #nav-toggle:checked ~ #nav-bar {
        width: 0;
    }

    #nav-toggle:checked ~ section {
        margin-left: 0;
    }
}

/* Custom Table Styling */
.custom-table {
    border-collapse: collapse;
    width: 100%;
    margin-top: 20px;
    background-color: #f0faff;
    color: #003366;
}

.custom-table th, .custom-table td {
    padding: 12px;
    border: 1px solid #d0e4f1;
    text-align: center;
    vertical-align: middle;
}

.custom-table th {
    background-color: #e6f4ff;
    color: #003366;
    font-weight: bold;
}

.custom-table tr:nth-child(even) {
    background-color: #ffffff;
}

.table-container {
    max-height: 400px;
    overflow-y: auto;
    border: 1px solid #d0e4f1;
    border-radius: 12px;
}

/* Custom Styling for Plotted Images */
.img-fluid {
    max-width: 100%;
    height: auto;
    border: 2px solid #d0e4f1;
    border-radius: 12px;
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    background-color: #f0faff;
    padding: 5px;
}

.img-fluid:hover {
    transform: scale(1.05);
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.25);
    border-color: #80cfff;
    background-color: #e6f4ff;
}

/* General Styles for Dashboard Cards */
.dashboard-card {
    background: linear-gradient(135deg, #ffffff 0%, #e0f7fa 100%);
    border-radius: 15px;
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    overflow: hidden;
    position: relative;
    padding: 20px;
    margin: 20px 0;
    animation: fadeInUp 0.6s ease-out;
}

.dashboard-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
}

/* Key-Value Pair Styles in Dashboard Cards */
.dashboard-item {
    margin-bottom: 15px;
    border-bottom: 1px solid #e0f7fa;
    padding-bottom: 10px;
}

.dashboard-key {
    font-size: 1.2rem;
    color: #003366;
    font-weight: bold;
    margin-bottom: 5px;
}

.dashboard-value {
    font-size: 1.3rem;
    color: #333;
    font-weight: normal;
}

/* Primary Button Styling */
.btn-primary {
    background-color: #007bff;
    border-color: #007bff;
    font-size: 1rem;
    padding: 12px 24px;
    border-radius: 10px;
    transition: background-color 0.3s ease, transform 0.3s ease;
}

.btn-primary:hover {
    background-color: #0056b3;
    border-color: #004494;
    transform: scale(1.05);
}

/* Animation Keyframes */
@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
 """