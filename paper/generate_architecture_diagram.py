import os
try:
    import graphviz
except ImportError:
    print("Please install graphviz: pip install graphviz")
    print("You may also need to install the system package (e.g., 'brew install graphviz' on Mac).")
    exit(1)

def create_diagram():
    # Configure graph attributes for NeurIPS style (Times New Roman-like, clean borders)
    dot = graphviz.Digraph(
        name="ICPA Diagram",
        format='pdf',
        graph_attr={
            'rankdir': 'LR',
            'fontname': 'Times-Roman',
            'fontsize': '14',
            'splines': 'ortho',
            'nodesep': '0.6',
            'ranksep': '1.0',
            'compound': 'true',
            'pad': '0.5'
        },
        node_attr={
            'fontname': 'Times-Roman',
            'fontsize': '12',
            'shape': 'box',
            'style': 'rounded,filled',
            'fillcolor': 'white',
            'color': 'black',
            'width': '1.8'
        },
        edge_attr={
            'fontname': 'Times-Roman',
            'fontsize': '10',
            'color': '#4a4a4a',
            'penwidth': '1.5'
        }
    )

    # --- Global Colors ---
    c_panel_bg = '#fcfbf8'
    c_panel_border = '#8b5a2b'
    c_model = '#fff8e1'        # Light yellow for models (Transfomers/LLMs)
    c_data = '#f0f4c3'         # Light green for data
    c_output = '#e3f2fd'       # Light blue for representations/outputs
    c_eval = '#fce4ec'         # Light pink for eval

    # panel A
    with dot.subgraph(name='cluster_a') as a:
        a.attr(
            label=" (a) Perception: Semantic Feature Fields & Instance Segmentation ",
            style='dashed,rounded',
            color=c_panel_border,
            fillcolor=c_panel_bg,
            penwidth='2'
        )
        
        a.node('raw_imgs', 'Raw UAV Data\n(Pre/Post Defol.)', fillcolor=c_data, shape='folder')
        
        a.node('dinov2', 'DINOv2 (ViT-L/14)\nVision Transformer', fillcolor=c_model)
        a.node('sam2', 'SAM 2\nSegment Anything', fillcolor=c_model)
        
        a.node('patch_embed', 'Semantic Embeddings\n(Dense patch-level info)', fillcolor=c_output)
        a.node('masks', 'Segmentation Masks\n(Robust instances)', fillcolor=c_output)
        
        a.node('unified', 'Unified Semantic Space\n(Dense Correspondences)', fillcolor=c_output, shape='Mrecord')

        a.edge('raw_imgs', 'dinov2')
        a.edge('raw_imgs', 'sam2')
        a.edge('dinov2', 'patch_embed')
        a.edge('sam2', 'masks')
        a.edge('patch_embed', 'unified')
        a.edge('masks', 'unified')

    # panel B
    with dot.subgraph(name='cluster_b') as b:
        b.attr(
            label=" (b) Geometry: Semantic 3D Reconstruction & Morphology ",
            style='dashed,rounded',
            color=c_panel_border,
            fillcolor=c_panel_bg,
            penwidth='2'
        )
        
        b.node('sba', 'Semantic Bundle\nAdjustment (SBA)', fillcolor=c_model)
        b.node('triangulate', 'Multi-View\nTriangulation', fillcolor=c_model)
        b.node('pcd', '3D Point Cloud\n(Cotton Bolls)', fillcolor=c_output, shape='cylinder')
        b.node('morph', 'Morphology Extraction\n(PCA, Convex Hull)', fillcolor=c_model)
        
        b.node('entity_graph', 'Context Entity Graph\n- Volume CV\n- Diameter\n- Visibility Score', fillcolor=c_output, shape='note')

        b.edge('sba', 'triangulate')
        b.edge('triangulate', 'pcd')
        b.edge('pcd', 'morph')
        b.edge('morph', 'entity_graph')

    # panel C
    with dot.subgraph(name='cluster_c') as c:
        c.attr(
            label=" (c) Cognition: Agronomist-in-the-Loop Reasoning ",
            style='dashed,rounded',
            color=c_panel_border,
            fillcolor=c_panel_bg,
            penwidth='2'
        )
        
        c.node('mod_f', 'Frames (F)', fillcolor=c_data)
        c.node('mod_c', 'Entity Graph (C)', fillcolor=c_data)
        c.node('mod_t', 'Text Report (T)', fillcolor=c_data)
        
        c.node('prompt', 'Domain Knowledge\nPrompt Templates', fillcolor=c_data)
        
        c.node('llm', 'Multimodal LLM\n(Frontier & Open-Weight)', fillcolor=c_model, shape='box3d')
        
        c.node('rec_stage', 'Growth Trajectory', fillcolor=c_output)
        c.node('rec_plan', 'Action Recommendations\n(PGR/Harvest Aid)', fillcolor=c_output)
        
        c.node('eval', 'Evaluation Metrics\n- Expert Agreement (%)\n- JSON Schema Rate\n- Hallucination-Free', fillcolor=c_eval)

        # invisible node to align inputs
        c.edge('mod_f', 'llm')
        c.edge('mod_c', 'llm')
        c.edge('mod_t', 'llm')
        c.edge('prompt', 'llm')
        
        c.edge('llm', 'rec_stage')
        c.edge('llm', 'rec_plan')
        
        c.edge('rec_plan', 'eval')

    # Inter-cluster edges (thick, dashed red like the reference)
    dot.edge('unified', 'sba', ltail='cluster_a', lhead='cluster_b', 
             color='#d32f2f', style='dashed', penwidth='3', xlabel='Alignment')
    
    dot.edge('entity_graph', 'mod_c', ltail='cluster_b', lhead='cluster_c', 
             color='#d32f2f', style='dashed', penwidth='3', xlabel='Graph Context')
             
    dot.edge('raw_imgs', 'mod_f', 
             color='#4a4a4a', style='dotted', penwidth='1')

    # Output paths
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'architecture_diagram'))
    dot.render(output_path, cleanup=True)
    print(f"Diagram successfully generated at: {output_path}.pdf")

if __name__ == "__main__":
    create_diagram()
