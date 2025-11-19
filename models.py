import torch
import torch.nn as nn
import timm

class EfficientNetWithConvStem(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super(EfficientNetWithConvStem, self).__init__()
        
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.feature_extraction(x)
        features = self.backbone(x)
        return self.classifier(features)


class EfficientNetWithMultiScale(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super(EfficientNetWithMultiScale, self).__init__()
        
        self.scale_large = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.scale_medium = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.scale_small = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        feat_large = self.scale_large(x)
        feat_medium = self.scale_medium(x)
        feat_small = self.scale_small(x)
        
        multi_scale = torch.cat([feat_large, feat_medium, feat_small], dim=1)
        fused = self.fusion(multi_scale)
        
        features = self.backbone(fused)
        return self.classifier(features)


class EfficientNetWithResidualStem(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super(EfficientNetWithResidualStem, self).__init__()
        
        self.feature_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.projection = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3)
        )
        
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        identity = x
        
        enhanced = self.feature_branch(x)
        enhanced = self.projection(enhanced)
        
        alpha = self.gate(enhanced)
        
        x = alpha * enhanced + (1 - alpha) * identity
        
        features = self.backbone(x)
        return self.classifier(features)


class EfficientNetWithDeepStem(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super(EfficientNetWithDeepStem, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        features = self.backbone(x)
        return self.classifier(features)


class EfficientNetWithAttentionStem(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super(EfficientNetWithAttentionStem, self).__init__()
        
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 32, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        self.projection = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        features = self.conv_stem(x)
        
        channel_att = self.channel_attention(features)
        features = features * channel_att
        
        avg_pool = torch.mean(features, dim=1, keepdim=True)
        max_pool, _ = torch.max(features, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        features = features * spatial_att
        
        x = self.projection(features)
        
        features = self.backbone(x)
        return self.classifier(features)

def compare_models():
    num_classes = 3
    batch_size = 2
    img_size = 224
    
    models = {
        'conv_stem': EfficientNetWithConvStem(num_classes),
        'multiscale': EfficientNetWithMultiScale(num_classes),
        'residual': EfficientNetWithResidualStem(num_classes),
        'deep_stem': EfficientNetWithDeepStem(num_classes),
        'attention': EfficientNetWithAttentionStem(num_classes),
    }
    
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"{name}: Params: {params}, Output: {output.shape}")

if __name__ == '__main__':
    compare_models()