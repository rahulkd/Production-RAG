# syntax=docker/dockerfile:1.7
#
# Three stages, mirroring the API image: resolve dependencies, build, then ship
# only the artifact. Nothing from the build toolchain reaches the final image.

FROM node:24-alpine AS deps

WORKDIR /app

# Copied on their own so this layer is only invalidated by a dependency change,
# not by every source edit.
COPY package.json package-lock.json ./

# BuildKit cache mount keeps the npm cache across builds without baking it into
# a layer. `npm ci` installs exactly what the lockfile pins.
RUN --mount=type=cache,target=/root/.npm \
    npm ci


FROM node:24-alpine AS builder

WORKDIR /app

ENV NEXT_TELEMETRY_DISABLED=1

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# next.config.ts sets output:"standalone", so this emits .next/standalone with a
# self-contained server and only the node_modules the build actually reached.
RUN npm run build


FROM node:24-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

RUN addgroup -g 1001 -S nodejs \
    && adduser -S nextjs -u 1001

COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD wget -qO- http://127.0.0.1:3000/api/health || exit 1

CMD ["node", "server.js"]
