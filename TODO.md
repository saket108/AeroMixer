# TODO: Convert Video Pipeline to Image + Text Multimodal

## Files Modified:

1. [x] ~~Analyze all files~~ - DONE
2. [x] common.py - Add image multimodal support (COMPLETED)
   - Added: ImageTextFusion, ImageStem2D classes
3. [x] nonlocal_helper.py - Add 2D Nonlocal block (COMPLETED)
   - Added: Nonlocal2D class
4. [x] resnet_helper.py - Add 2D ResNet components (COMPLETED)
   - Added: BasicTransform2D, BottleneckTransform2D, ResBlock2D, ResStage2D
5. [ ] stem_helper.py - Convert video stem to image stem (IN PROGRESS)
6. [ ] i3d.py - Convert to image backbone
7. [ ] slowfast.py - Convert to image backbone
8. [ ] video_model_builder.py - Add image-only model builders
9. [ ] vit_utils.py - Ensure 2D mode works properly
10. [ ] backbone.py - Register new image backbones

## Summary:
- Completed: common.py, nonlocal_helper.py, resnet_helper.py
- In Progress: stem_helper.py
- Pending: Other backbone files
