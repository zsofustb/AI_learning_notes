# Linux安装codex cli

### 安装nvm
```
git clone https://ghfast.top/https://github.com/nvm-sh/nvm.git .nvm
cd .nvm && . nvm.sh
```

### nvm安装/管理nodejs和npm
```
nvm install node // 安装最新版nodejs
nvm install 14   //安装14.xx版本nodejs【nvm install 14.19.0 安装指定版本的node】
nvm list         //显示已安装nodejs
nvm use 14       //使用14.xx版本nodejs【nvm use 14.19.0 指定使用的node版本】
```

### npm安装codex cli
```
npm install -g @openai/codex --registry=https://registry.npmmirror.com
```
