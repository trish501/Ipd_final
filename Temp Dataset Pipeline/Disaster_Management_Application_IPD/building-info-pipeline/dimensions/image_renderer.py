import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import rasterio
import pyproj
from shapely.ops import transform
import numpy as np
import math

def render_annotated_image(rgb_tif_path: str, buildings: list, institution_summary, imagery_summary, out_png_path: str, location_str: str, warnings: list = None, campus_geom = None, lat: float = 0.0, lon: float = 0.0):
    _render_image(rgb_tif_path, buildings, institution_summary, imagery_summary, out_png_path, location_str, enhanced=False, warnings=warnings, campus_geom=campus_geom, lat=lat, lon=lon)
    
    out_enhanced_path = out_png_path.replace(".png", "_enhanced.png")
    _render_image(rgb_tif_path, buildings, institution_summary, imagery_summary, out_enhanced_path, location_str, enhanced=True, warnings=warnings, campus_geom=campus_geom, lat=lat, lon=lon)

def _render_image(rgb_tif_path: str, buildings: list, institution_summary, imagery_summary, out_png_path: str, location_str: str, enhanced: bool, warnings: list, campus_geom, lat: float, lon: float):
    with rasterio.open(rgb_tif_path) as src:
        crs = src.crs
        
        img = src.read()
        img = img.astype(float)
        img = np.clip(img / 255.0, 0, 1)
        img = np.transpose(img, (1, 2, 0))
        
        fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
        fig.patch.set_facecolor('black')
        
        if enhanced:
            ax.imshow(img, interpolation='bicubic')
            ax.text(0.5, 0.94, "VISUALIZATION ONLY - ENHANCED INTERPOLATION", transform=ax.transAxes, 
                    fontsize=14, color='#FFD700', weight='bold', ha='center', va='top', 
                    path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])
        else:
            ax.imshow(img, interpolation='nearest')
            
        ax.axis('off')
        
        # Add Top Title
        ax.text(0.5, 0.98, f"INSTITUTION MEASUREMENT - {location_str.upper()}", transform=ax.transAxes, 
                fontsize=20, color='white', weight='bold', ha='center', va='top', 
                path_effects=[path_effects.withStroke(linewidth=4, foreground='black')])
        
        wgs84 = pyproj.CRS("EPSG:4326")
        proj_crs = pyproj.CRS(crs)
        transformer = pyproj.Transformer.from_crs(wgs84, proj_crs, always_xy=True)
        
        transform_matrix = ~src.transform
        
        # 1. Render Campus Highlight
        if campus_geom:
            geom_crs = transform(transformer.transform, campus_geom)
            if geom_crs.geom_type == 'Polygon' or geom_crs.geom_type == 'MultiPolygon':
                # Handle multipolygons by iterating
                polys = [geom_crs] if geom_crs.geom_type == 'Polygon' else geom_crs.geoms
                for poly in polys:
                    ext_coords = poly.exterior.coords
                    pix_coords = [transform_matrix * (x, y) for x, y in ext_coords]
                    # Translucent fill
                    poly_patch = patches.Polygon(pix_coords, closed=True, facecolor='#90EE90', edgecolor='#90EE90', alpha=0.10, linewidth=1.5)
                    ax.add_patch(poly_patch)
                    # Sharp border over it
                    border_patch = patches.Polygon(pix_coords, closed=True, facecolor='none', edgecolor='#90EE90', linewidth=1.5, alpha=0.8)
                    ax.add_patch(border_patch)
                    
                    # Interims
                    for inter in poly.interiors:
                        i_coords = [transform_matrix * (x, y) for x, y in inter.coords]
                        i_line = patches.Polygon(i_coords, closed=True, facecolor='none', edgecolor='#90EE90', linewidth=1.5, alpha=0.8)
                        ax.add_patch(i_line)

        # 2. Render Buildings
        num_buildings = len(buildings)
        # Adaptive font sizes based on density
        if num_buildings < 20:
            font_size = 14
            label_font_size = 16
            line_width = 2.0
        elif num_buildings < 50:
            font_size = 10
            label_font_size = 12
            line_width = 1.5
        elif num_buildings < 100:
            font_size = 8
            label_font_size = 10
            line_width = 1.0
        else:
            font_size = 6
            label_font_size = 8
            line_width = 0.5
        # Determine which buildings get full text labels (top 15 by area) to avoid clutter
        sorted_buildings = sorted(buildings, key=lambda x: x.footprint_area_sq_m, reverse=True)
        buildings_to_label = {x.building_id for x in sorted_buildings[:15]}
        
        for i, b in enumerate(buildings):
            if not b.geometry_wgs84:
                continue
                
            geom_crs = transform(transformer.transform, b.geometry_wgs84)
            
            if geom_crs.geom_type == 'Polygon':
                ext_coords = geom_crs.exterior.coords
                pix_coords = [transform_matrix * (x, y) for x, y in ext_coords]
                # Outline every building
                poly_patch = patches.Polygon(pix_coords, closed=True, fill=False, edgecolor='white', linewidth=line_width, alpha=0.9)
                ax.add_patch(poly_patch)
                
                # To prevent massive clutter, only draw text and arrows for the top 15 largest buildings
                if b.building_id not in buildings_to_label:
                    continue
                
                
                # Draw dimension lines and text for ALL buildings
                mrr = geom_crs.minimum_rotated_rectangle
                if mrr.geom_type == 'Polygon':
                    coords = list(mrr.exterior.coords)
                    if len(coords) >= 4:
                        edges = []
                        for j in range(4):
                            p1, p2 = coords[j], coords[j+1]
                            dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
                            edges.append((dist, p1, p2))
                        edges.sort(key=lambda x: x[0], reverse=True)
                        
                        long_edge = edges[0]
                        short_edge = edges[2]
                        
                        p1_px = transform_matrix * long_edge[1]
                        p2_px = transform_matrix * long_edge[2]
                        ax.annotate('', xy=p2_px, xytext=p1_px, arrowprops=dict(arrowstyle='<|-|>', color='white', lw=line_width, alpha=0.8))
                        
                        mid_x = (p1_px[0] + p2_px[0]) / 2
                        mid_y = (p1_px[1] + p2_px[1]) / 2
                        ax.text(mid_x, mid_y, f"{b.length_m:.1f}m", color='white', fontsize=font_size, ha='center', va='center',
                                path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')])
                        
                        s1_px = transform_matrix * short_edge[1]
                        s2_px = transform_matrix * short_edge[2]
                        ax.annotate('', xy=s2_px, xytext=s1_px, arrowprops=dict(arrowstyle='<|-|>', color='white', lw=line_width, alpha=0.8))
                        
                        smid_x = (s1_px[0] + s2_px[0]) / 2
                        smid_y = (s1_px[1] + s2_px[1]) / 2
                        ax.text(smid_x, smid_y, f"{b.width_m:.1f}m", color='white', fontsize=font_size, ha='center', va='center',
                                path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')])
                
                c_lon, c_lat = b.centroid['longitude'], b.centroid['latitude']
                cx, cy = transformer.transform(c_lon, c_lat)
                pcx, pcy = transform_matrix * (cx, cy)
                
                label_text = f"{b.building_id}\n{b.length_m:.1f}×{b.width_m:.1f}m\n{b.footprint_area_sq_m:,.1f}m²"
                ax.text(pcx, pcy, label_text, color='white', fontsize=label_font_size, ha='center', va='center', weight='bold',
                        bbox=dict(facecolor='green', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.2'))

        # 3. Summary Box
        warn_str = ""
        if warnings:
            warn_str = "\n\nWARNINGS:\n" + "\n".join([f"- {w}" for w in warnings])
            
        summary_text = (
            "INSTITUTION SUMMARY\n"
            "─────────────────────────\n"
            f"Institution: {location_str}\n"
            f"Location: {lat:.4f}° N, {lon:.4f}° E\n"
            f"Buildings (Displayed): {len(buildings)}\n"
            f"Total Buildings Detected: {institution_summary.selected_buildings}\n"
            "─────────────────────────\n"
            "Total Building Area:\n"
            f"{institution_summary.total_building_footprint_area_sq_m:,.1f} m²\n"
            f"{institution_summary.total_building_footprint_area_sq_ft:,.1f} ft²\n"
            "─────────────────────────\n"
            f"Imagery: {imagery_summary.provider}\n"
            f"Acquisition: {imagery_summary.acquisition_datetime[:10]}\n"
            f"Native Resolution: {imagery_summary.native_resolution_m:.1f}m"
            f"{warn_str}"
        )
        props = dict(boxstyle='round,pad=0.5', facecolor='#111111', edgecolor='#90EE90', alpha=0.8)
        
        ax.text(0.02, 0.02, summary_text, transform=ax.transAxes, fontsize=10, color='white', 
                verticalalignment='bottom', bbox=props, family='monospace', linespacing=1.5)
                
        # 4. Legend Box
        legend_elements = [
            patches.Patch(facecolor='#90EE90', alpha=0.3, edgecolor='#90EE90', label='Institution Boundary'),
            patches.Patch(facecolor='none', edgecolor='white', label='Building Footprint'),
            plt.Line2D([0], [0], color='white', lw=1, marker='<', markersize=5, label='Length (m)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', facecolor='#111111', edgecolor='white', 
                  labelcolor='white', framealpha=0.8, borderpad=1)
        
        plt.tight_layout(pad=0)
        plt.savefig(out_png_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
