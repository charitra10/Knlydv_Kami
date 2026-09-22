#include "helper.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace bot {
using namespace unswbc;

constexpr int MAX_CELLS = 4096, INF = 100000;
const auto DIRS = Direction::get_direction_list();

inline int di(Direction d) {
    for (int i = 0; i < 4; ++i)
        if (DIRS[i] == d)
            return i;
    return 0;
}

enum class Mode { Disperse, Forage, Ambush, Escape, Portal, Reside, Return, Feed, Split, Kill, Trapped };

inline const char *name(Mode m) {
    static const char *names[] = {"disperse", "forage", "ambush", "escape", "portal", "reside",
                                  "return",   "feed",   "split",  "kill",   "trapped"};
    return names[static_cast<int>(m)];
}

struct Action {
    std::vector<Direction> moves;
    int child = 0;
    Mode mode = Mode::Forage;
    bool intentional_death = false;
};

struct Cell {
    int seen = -1, visited = -10000;
    std::array<int, 4> edge{{-2, -2, -2, -2}}; // unknown=-2, kelp=-1, open=0, portal=id+1
};

struct Portal {
    int id;
    std::vector<std::pair<Position, int>> ends;
    int occupied_until = -1;
};

struct AlphaTrack {
    int id, seen;
    Position p;
};

struct DragonState {
    bool initialized = false, alpha = false, growing = false, resident = false, dispersing = false;
    int born = 0, last_food = 0, sector = 0, home_portal = -1, pending_portal = -1;
    Position sector_target;
    Position return_tile;
    std::array<Cell, MAX_CELLS> cells{};
    std::vector<Portal> portals;
    std::vector<AlphaTrack> alphas;
    std::vector<DragonPart> previous_heads;
};

class Brain {
    Controller &c;
    Game &g;
    int w, h, round = 0;
    Position here;
    std::vector<DragonPart> friends, enemies;

    // Pre-allocated scratch buffers on heap to guarantee zero stack overflow in WebAssembly
    std::array<int, MAX_CELLS> distance{}, first{}, predecessor{}, arrival{};
    mutable std::array<int, MAX_CELLS> risk_cache{};
    mutable std::array<int, MAX_CELLS> danger_depth{};
    mutable std::array<bool, MAX_CELLS> space_seen{};
    mutable std::array<int, MAX_CELLS> remembered_first{};
    mutable std::array<Position, MAX_CELLS> remembered_q{};

  public:
    DragonState s;

    Brain(Controller &controller, Game &game_state) : c(controller), g(game_state) {
        std::tie(w, h) = g.get_map_size();
    }

    int index(Position p) const {
        return p.y * w + p.x;
    }

    Position wrap(Position p) const {
        return {(p.x % w + w) % w, (p.y % h + h) % h};
    }

    Position step(Position p, int d) const {
        auto [x, y] = DIRS[d].get_offset();
        return wrap({p.x + x, p.y + y});
    }

    int delta(int a, int b, int size) const {
        int v = (b - a + size) % size;
        if (v > size / 2)
            v -= size;
        return v;
    }

    int dist(Position a, Position b) const {
        return std::abs(delta(a.x, b.x, w)) + std::abs(delta(a.y, b.y, h));
    }

    int chebyshev(Position a, Position b) const {
        return std::max(std::abs(delta(a.x, b.x, w)), std::abs(delta(a.y, b.y, h)));
    }

    Portal &portal(int id) {
        for (auto &p : s.portals)
            if (p.id == id)
                return p;
        s.portals.push_back({id, {}, -1});
        return s.portals.back();
    }

    std::pair<Position, int> canonical(Position p, int d) const {
        if (d == 2)
            return {step(p, 2), 0};
        if (d == 1)
            return {step(p, 1), 3};
        return {p, d};
    }

    // A portal never means the ordinary adjacent board square.
    std::optional<Position> destination(Position p, int d) const {
        int e = s.cells[index(p)].edge[d];
        if (e < 0)
            return {};
        if (e == 0)
            return step(p, d);
        auto from = canonical(p, d);
        for (const auto &port : s.portals)
            if (port.id == e - 1 && port.ends.size() == 2) {
                if (port.ends[0] != from && port.ends[1] != from)
                    return {};
                auto to = port.ends[0] == from ? port.ends[1] : port.ends[0];
                return (d == 0 || d == 3) ? step(to.first, d) : to.first;
            }
        return {};
    }

    bool empty(Position p) const {
        auto t = c.get_tile(p);
        return t && !t->get_dragon();
    }

    void remember_alpha(int id, Position p) {
        for (auto &a : s.alphas)
            if (a.id == id) {
                a = {id, round, p};
                return;
            }
        s.alphas.push_back({id, round, p});
    }

    bool is_alpha(int id) const {
        if (std::max(w, h) < 20)
            return false;
        if (id <= 1)
            return true;
        for (auto a : s.alphas)
            if (a.id == id)
                return true;
        return false;
    }

    int threshold() const {
        int n = std::max(w, h);
        return n < 22 ? 460 : n <= 35 ? 320 : n <= 50 ? 200 : 100;
    }

    void initialize() {
        s.initialized = true;
        s.born = round;
        s.last_food = round;

        // Requirement: For maps < 20x20 make all bots kamikaze and no alpha needed
        if (std::max(w, h) < 20) {
            s.alpha = false;
        } else {
            // Only initial spawn dragon (id <= 1) is Alpha; all split children are Kamikaze
            s.alpha = (c.get_id() <= 1);
        }

        // Team-relative index so both Team A (even IDs) and Team B (odd IDs)
        // get evenly distributed across sectors 0, 1, 2, 3!
        s.sector = (c.get_id() / 2) % 4;

        // Symmetric fan-out toward team's corners/edges without torus-wrapping bugs
        bool is_left = (here.x < w / 2) || (here.x == w / 2 && c.get_team() == Team::A);
        int outer_x = is_left ? 0 : w - 1;
        int inner_x = is_left ? w / 4 : w - 1 - w / 4;

        switch (s.sector) {
        case 0: s.sector_target = {outer_x, 0}; break;
        case 1: s.sector_target = {outer_x, h - 1}; break;
        case 2: s.sector_target = {outer_x, h / 2}; break;
        default: s.sector_target = {inner_x, h / 2}; break;
        }
        s.dispersing = true;
    }

    void observe() {
        round = g.get_round_num();
        here = c.get_position();
        friends.clear();
        enemies.clear();
        risk_cache.fill(-1);

        auto cur_tile = c.get_tile(here);
        if (cur_tile && cur_tile->has_pearl())
            s.last_food = round;

        for (const auto &t : c.get_tiles()) {
            auto p = t.get_position();
            auto &cell = s.cells[index(p)];
            cell.seen = round;
            for (int d = 0; d < 4; ++d) {
                const auto &e = t.get_edge(DIRS[d]);
                cell.edge[d] = !e.is_passable() ? -1 : e.is_portal() ? e.get_portal_id() + 1 : 0;
                if (e.is_portal()) {
                    auto &port = portal(e.get_portal_id());
                    auto endpoint = canonical(p, d);
                    if (std::find(port.ends.begin(), port.ends.end(), endpoint) == port.ends.end())
                        port.ends.push_back(endpoint);
                }
            }
            auto part = t.get_dragon();
            if (part && part->is_head() && part->get_id() != c.get_id())
                (part->get_team() == c.get_team() ? friends : enemies).push_back(*part);
        }

        auto by_id = [](const DragonPart &a, const DragonPart &b) { return a.get_id() < b.get_id(); };
        std::sort(friends.begin(), friends.end(), by_id);
        std::sort(enemies.begin(), enemies.end(), by_id);

        if (!s.initialized)
            initialize();

        if (dist(here, s.sector_target) <= 2 || round - s.born > std::max(w, h))
            s.dispersing = false;

        s.cells[index(here)].visited = round;

        if (s.pending_portal >= 0) {
            if (s.resident && s.home_portal == s.pending_portal) {
                s.resident = false;
                s.home_portal = -1;
            } else {
                s.resident = true;
                s.home_portal = s.pending_portal;
                s.return_tile = here;
            }
            s.last_food = round;
            s.pending_portal = -1;
        }

        for (auto f : friends) {
            if (is_alpha(f.get_id()))
                remember_alpha(f.get_id(), f.position);
        }

        // Sonar message decoding
        for (auto msg : c.get_sonar_messages())
            if ((msg >> 24) == 0xA7u) {
                int id = static_cast<int>((msg >> 12) & 4095u);
                Position p{static_cast<int>((msg >> 6) & 63u), static_cast<int>(msg & 63u)};
                for (auto f : friends)
                    if (id <= 1 && f.get_id() == id && dist(f.position, p) <= 1)
                        remember_alpha(id, f.position);
            }

        s.previous_heads = friends;
        s.alphas.erase(std::remove_if(s.alphas.begin(), s.alphas.end(),
                                      [&](const AlphaTrack &a) { return round - a.seen > 24; }),
                       s.alphas.end());

        if (s.alpha && (round > threshold() || c.get_unit_count() > 43))
            s.growing = true;
    }

    bool portal_occupied(int id) const {
        for (const auto &t : c.get_tiles()) {
            auto part = t.get_dragon();
            if (!part || part->get_team() != c.get_team() || !part->is_head())
                continue;
            for (int d = 0; d < 4; ++d) {
                const auto &edge = t.get_edge(DIRS[d]);
                if (edge.is_portal() && edge.get_portal_id() == id &&
                    part->get_dir() == DIRS[d].get_opposite())
                    return true;
            }
        }
        return false;
    }

    void paths() {
        distance.fill(INF);
        first.fill(-1);
        predecessor.fill(-1);
        arrival.fill(-1);
        std::array<Position, 49> q{};
        int lo = 0, hi = 0;
        q[hi++] = here;
        distance[index(here)] = 0;
        while (lo < hi) {
            auto p = q[lo++];
            int pi = index(p);
            for (int d = 0; d < 4; ++d) {
                if (s.cells[pi].edge[d] != 0)
                    continue;
                auto n = step(p, d);
                int ni = index(n);
                auto t = c.get_tile(n);
                if (!t || distance[ni] != INF)
                    continue;
                auto part = t->get_dragon();
                if (part && !(part->get_team() != c.get_team() && part->is_head()))
                    continue;
                distance[ni] = distance[pi] + 1;
                first[ni] = pi == index(here) ? d : first[pi];
                predecessor[ni] = pi;
                arrival[ni] = d;
                if (!part)
                    q[hi++] = n;
            }
        }
    }

    int escape_count(Position p) const {
        int n = 0;
        for (int d = 0; d < 4; ++d) {
            int edge = s.cells[index(p)].edge[d];
            if (edge == 0 && empty(step(p, d)))
                ++n;
            else if (edge > 0 && !portal_occupied(edge - 1))
                ++n; // Portals are valid escapes!
        }
        return n;
    }

    int danger(Position p) const {
        if (risk_cache[index(p)] >= 0)
            return risk_cache[index(p)];
        int score = 0;
        for (auto e : enemies) {
            int visible_length = 0;
            bool partial = false;
            for (const auto &t : c.get_tiles())
                if (t.get_dragon() && t.get_dragon()->get_id() == e.get_id()) {
                    ++visible_length;
                    if (std::abs(delta(here.x, t.position.x, w)) == 3 ||
                        std::abs(delta(here.y, t.position.y, h)) == 3)
                        partial = true;
                }
            int budget =
                std::min(6, std::max(1, (partial ? std::max(visible_length, 4) : visible_length) - 1));

            // Use heap-allocated member danger_depth instead of stack array
            danger_depth.fill(-1);
            std::array<Position, 49> q{};
            int lo = 0, hi = 0;
            q[hi++] = e.position;
            danger_depth[index(e.position)] = 0;
            int reach = INF;
            while (lo < hi) {
                auto at = q[lo++];
                int n = danger_depth[index(at)];
                if (n >= budget)
                    continue;
                for (int d = 0; d < 4; ++d)
                    if (s.cells[index(at)].edge[d] == 0) {
                        auto to = step(at, d);
                        auto tile = c.get_tile(to);
                        if (!tile)
                            continue;
                        if (to == p) {
                            reach = std::min(reach, n + 1);
                            continue;
                        }
                        if (tile->get_dragon() || danger_depth[index(to)] >= 0)
                            continue;
                        danger_depth[index(to)] = n + 1;
                        q[hi++] = to;
                    }
            }
            if (reach < INF)
                score += 400 - 40 * reach;
            else if (dist(p, e.position) <= 3)
                score += 4;
        }
        risk_cache[index(p)] = score;
        return score;
    }

    int space(Position start) const {
        space_seen.fill(false);
        std::array<Position, 49> q{};
        int lo = 0, hi = 0;
        q[hi++] = start;
        space_seen[index(start)] = true;
        while (lo < hi) {
            auto p = q[lo++];
            for (int d = 0; d < 4; ++d)
                if (s.cells[index(p)].edge[d] == 0) {
                    auto n = step(p, d);
                    if (!space_seen[index(n)] && empty(n)) {
                        space_seen[index(n)] = true;
                        q[hi++] = n;
                    }
                }
        }
        return hi;
    }

    bool incoming() const {
        for (auto e : enemies)
            if (dist(e.position, here) <= 4 &&
                dist(step(e.position, di(e.get_dir())), here) < dist(e.position, here))
                return true;
        return false;
    }

    std::vector<Direction> route(Position target) const {
        std::vector<Direction> out;
        int at = index(target);
        while (at != index(here) && at >= 0 && arrival[at] >= 0) {
            out.push_back(DIRS[arrival[at]]);
            at = predecessor[at];
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    bool is_step_safe(int d, bool allow_enemy_head = false) const {
        int opp_dir = (di(c.get_dir()) + 2) % 4;
        if (d == opp_dir)
            return false;

        int edge = s.cells[index(here)].edge[d];
        if (edge < 0)
            return false; // impassable / kelp

        if (edge > 0) {
            // Portal: safe if not occupied by a friendly exit, and far side (if known) is unblocked
            if (portal_occupied(edge - 1))
                return false;
            auto exit = destination(here, d);
            if (exit) {
                auto t = c.get_tile(*exit);
                if (t) {
                    auto dragon = t->get_dragon();
                    if (dragon) {
                        if (dragon->get_id() == c.get_id()) return false;
                        if (dragon->get_team() == c.get_team()) return false;
                        if (!dragon->is_head()) return false;
                        if (!allow_enemy_head) return false;
                    }
                }
            }
            return true;
        }

        // Normal edge
        auto dest = step(here, d);
        auto t = c.get_tile(dest);
        if (t) {
            auto dragon = t->get_dragon();
            if (dragon) {
                if (dragon->get_id() == c.get_id()) return false;
                if (dragon->get_team() == c.get_team()) return false;
                if (!dragon->is_head()) return false;
                if (!allow_enemy_head) return false;
            }
        }
        return true;
    }

    std::optional<Action> guaranteed_kill() const {
        if (s.alpha || c.get_unit_count() <= 1)
            return {};

        // 1. Immediate 1-step adjacent collision
        for (const auto &e : enemies) {
            if (dist(here, e.position) == 1) {
                for (int d = 0; d < 4; ++d) {
                    if (step(here, d) == e.position) {
                        int edge = s.cells[index(here)].edge[d];
                        if (edge == 0) {
                            return Action{{DIRS[d]}, 0, Mode::Kill, true};
                        }
                    }
                }
            }
        }

        // 2. 2-step sprint kill (strictly distance 2 to avoid blind reckless sprints)
        if (c.get_length() >= 3) {
            for (const auto &e : enemies) {
                if (distance[index(e.position)] != 2)
                    continue;
                auto candidate = route(e.position);
                if (candidate.size() != 2)
                    continue;
                Position inter = step(here, di(candidate[0]));
                auto t_inter = c.get_tile(inter);
                if (!t_inter || t_inter->get_dragon())
                    continue;
                return Action{candidate, 0, Mode::Kill, true};
            }
        }
        return {};
    }

    bool claimed(Position p, int mydistance) const {
        for (auto f : friends) {
            int k = dist(f.position, p);
            if (k < mydistance || (k == mydistance && f.get_id() < c.get_id()))
                return true;
        }
        return false;
    }

    std::optional<Position> food_target(bool future) const {
        std::optional<Position> best;
        double bestscore = -1e20;
        for (const auto &t : c.get_tiles()) {
            auto p = t.get_position();
            int n = distance[index(p)];
            if (n < 1 || n > 10 || t.get_dragon())
                continue;
            if (future ? !(t.get_pearl_time() >= 1 && t.get_pearl_time() <= 2) : !t.has_pearl())
                continue;
            if (claimed(p, n))
                continue;

            int exits = escape_count(p);
            if (s.alpha && (danger(p) >= 100 || exits < (c.get_length() >= 12 ? 3 : 2)))
                continue;

            // Cluster richness score from kamikaze
            double cluster = 0;
            for (const auto &other : c.get_tiles())
                if (other.has_pearl()) {
                    int k = chebyshev(p, other.get_position());
                    if (k <= 2)
                        cluster += (k == 0 ? 0 : 1.5 / k);
                }

            // High base score so pearls are strongly prioritized
            double score = (30.0 + 10.0 * cluster) / (n + 0.5) - 0.08 * danger(p);

            // Avoid suppressing upcoming pearls
            if (future && n < t.get_pearl_time())
                score -= 10;

            if (!best || score > bestscore || (score == bestscore && index(p) < index(*best))) {
                best = p;
                bestscore = score;
            }
        }
        return best;
    }

    struct PortalPlan {
        Position approach;
        int direction, id, cost;
    };

    std::optional<PortalPlan> portal_target(bool force) const {
        std::optional<PortalPlan> best;
        for (const auto &t : c.get_tiles())
            for (int d = 0; d < 4; ++d) {
                auto p = t.get_position();
                int e = s.cells[index(p)].edge[d];
                if (e <= 0 || distance[index(p)] == INF || (p != here && t.get_dragon()))
                    continue;
                if (!force && portal_occupied(e - 1))
                    continue;
                if (!force && s.resident && (round - s.last_food < 35 || e - 1 != s.home_portal))
                    continue;
                auto exit = destination(p, d);
                if (exit) {
                    auto tile = c.get_tile(*exit);
                    if (tile && tile->get_dragon())
                        continue;
                }
                int cost = distance[index(p)] + 1;
                if (!best || cost < best->cost || (cost == best->cost && e - 1 < best->id))
                    best = PortalPlan{p, d, e - 1, cost};
            }
        return best;
    }

    std::optional<Position> feed_target() const {
        std::optional<Position> target;
        int best = INF;
        for (auto f : friends)
            if (is_alpha(f.get_id())) {
                int k = dist(here, f.position);
                if (k < best) {
                    best = k;
                    target = f.position;
                }
            }
        if (!target)
            for (auto a : s.alphas) {
                int k = dist(here, a.p) + round - a.seen;
                if (k < best) {
                    best = k;
                    target = a.p;
                }
            }
        return target;
    }

    int remembered_direction(Position target) const {
        remembered_first.fill(-1);
        int lo = 0, hi = 0;
        remembered_q[hi++] = here;
        remembered_first[index(here)] = 4;
        while (lo < hi) {
            auto p = remembered_q[lo++];
            for (int d = 0; d < 4; ++d)
                if (s.cells[index(p)].edge[d] == 0) {
                    auto n = step(p, d);
                    auto t = c.get_tile(n);
                    if (remembered_first[index(n)] >= 0 || (t && t->get_dragon()) || s.cells[index(n)].seen < 0)
                        continue;
                    remembered_first[index(n)] = p == here ? d : remembered_first[index(p)];
                    if (n == target)
                        return remembered_first[index(n)];
                    remembered_q[hi++] = n;
                }
        }
        return -1;
    }

    Action decide() {
        observe();
        paths();

        bool feed = !s.alpha && round >= 440;
        auto alpha = feed ? feed_target() : std::optional<Position>{};
        if (feed && alpha && dist(here, *alpha) == 1) {
            auto t = c.get_tile(*alpha);
            if (t && t->get_dragon() && t->get_dragon()->get_team() == c.get_team() &&
                t->get_dragon()->is_head() && is_alpha(t->get_dragon()->get_id())) {
                // Self destruct into neck/body
                int opp = (di(c.get_dir()) + 2) % 4;
                return {{DIRS[opp]}, 0, Mode::Feed, true};
            }
        }

        // 1. Splitting check: split whenever possible to grow army
        if (c.can_split(c.get_length() - 2)) {
            bool do_split = false;
            if (!s.alpha) {
                if (c.get_length() >= 4 && c.get_unit_count() < 64 && round < 440)
                    do_split = true;
            } else {
                if (c.get_length() >= 4 && !s.growing)
                    do_split = true;
            }
            if (do_split)
                return {{}, c.get_length() - 2, Mode::Split, false};
        }

        // 2. Predatory kill check (only if team has backup dragons)
        if (!feed && c.get_unit_count() > 1) {
            if (auto kill = guaranteed_kill())
                return *kill;
        }

        // Disallow 180-degree neck reversal to avoid accidental Hit Self deaths
        int opp_dir = (di(c.get_dir()) + 2) % 4;
        bool allow_enemy_head = !s.alpha && c.get_unit_count() > 1;
        std::vector<int> legal;
        for (int d = 0; d < 4; ++d) {
            if (is_step_safe(d, allow_enemy_head))
                legal.push_back(d);
        }

        bool threatened = s.alpha && (incoming() || danger(here) >= 100), perpendicular = false;
        for (int d : legal)
            if ((d % 2) != (di(c.get_dir()) % 2) && danger(step(here, d)) < 100)
                perpendicular = true;

        // Emergency split if trapped
        if (c.can_split(c.get_length() - 2) && (legal.empty() || (threatened && !perpendicular)))
            return {{}, c.get_length() - 2, Mode::Split, false};

        auto pearls = food_target(false);
        auto future = pearls ? std::optional<Position>{} : food_target(true);
        auto port = (!s.alpha) ? portal_target(false) : std::optional<PortalPlan>{};

        bool first_four_turns = round - s.born < 4;
        bool early_portal = !s.alpha && first_four_turns && !s.resident && !pearls && port.has_value();
        bool return_portal = s.resident && round - s.last_food >= 35 && port.has_value();
        bool normal_portal = !s.alpha && !s.resident && !pearls && !future && port.has_value();

        bool use_portal = early_portal || return_portal || normal_portal;

        std::optional<Position> target = pearls ? pearls : future;
        Mode mode = s.resident ? Mode::Reside : Mode::Forage;

        if (feed) {
            if (alpha)
                target = alpha;
            mode = Mode::Feed;
            use_portal = false;
        }

        if (return_portal && !port && !feed) {
            target = s.return_tile;
            mode = Mode::Return;
        }

        if (use_portal) {
            mode = return_portal ? Mode::Return : Mode::Portal;
            if (port->approach == here) {
                if (is_step_safe(port->direction, allow_enemy_head)) {
                    s.pending_portal = port->id;
                    return {{DIRS[port->direction]}, 0, mode, false};
                }
            } else {
                target = port->approach;
            }
        }

        if (threatened)
            mode = Mode::Escape;

        if (!target && !feed && !threatened && round - s.born < 6 && dist(here, s.sector_target) > 2) {
            target = s.sector_target;
            mode = Mode::Disperse;
        }

        // Opposing vectors ambush: cut into enemy head path or shadow laterally
        if (!s.alpha && !feed && !use_portal) {
            for (auto e : enemies) {
                int ed = di(e.get_dir());
                if (dist(step(e.position, ed), here) >= dist(e.position, here))
                    continue; // not opposing / facing us

                // Same direction pursuit avoidance: don't chase rear moving away
                if (e.get_dir() == c.get_dir())
                    continue;

                auto aim = step(e.position, ed); // cut directly into its head destination
                if (empty(aim) && distance[index(aim)] < INF) {
                    target = aim;
                    mode = Mode::Ambush;
                    break;
                }
                auto lateral = step(aim, (ed + ((c.get_id() % 2) ? 1 : 3)) % 4);
                if (empty(lateral) && distance[index(lateral)] < INF) {
                    target = lateral;
                    mode = Mode::Ambush;
                    break;
                }
            }
        }

        int target_direction = target ? first[index(*target)] : -1;
        if (target && target_direction < 0)
            target_direction = remembered_direction(*target);

        // For Kamikaze: direct BFS path execution for targets (pearls or portal approach) when step is safe
        if (!s.alpha && target && target_direction >= 0 &&
            is_step_safe(target_direction, allow_enemy_head)) {
            return {{DIRS[target_direction]}, 0, mode, false};
        }

        double best = -1e30;
        int chosen = -1;

        for (int d : legal) {
            int edge = s.cells[index(here)].edge[d];
            if (edge > 0)
                continue;
            auto n = step(here, d);
            auto tile = c.get_tile(n);
            int exits = escape_count(n), area = space(n), risk = danger(n);

            double score = 2.0 * std::min(area, 10) + 4.0 * exits - risk * (s.alpha ? 2.5 : 0.05);

            if (exits == 0)
                score -= 800;
            else if (exits == 1)
                score -= s.alpha ? 80 : 15; // Lower penalty for Kamikaze so it doesn't fear eating pearls

            if (threatened && perpendicular && (d % 2) == (di(c.get_dir()) % 2))
                score -= 400;

            if (target) {
                if (target_direction == d)
                    score += 80;
                else
                    score += 15 * (dist(here, *target) - dist(n, *target));
            }

            // Strong bonus for eating pearls directly
            if (tile && tile->has_pearl() &&
                (!s.alpha || (risk < 100 && exits >= (c.get_length() >= 12 ? 3 : 2))))
                score += 70;

            // Avoid suppressing pearls that spawn next turn
            if (tile && !tile->has_pearl() && tile->get_pearl_time() == 1)
                score -= 15;

            score += std::min(round - s.cells[index(n)].visited, 30) * 0.4;
            if (d == di(c.get_dir()))
                score += 2; // slight forward momentum

            // Teammate repulsion to prevent clumping
            for (auto f : friends) {
                int d_team = dist(n, f.position);
                if (d_team < 3)
                    score -= 5 * (3 - d_team);
            }

            if (score > best) {
                best = score;
                chosen = d;
            }
        }

        if (chosen >= 0)
            return {{DIRS[chosen]}, 0, mode, false};

        // 1. Any safe portal edge right here
        for (int d = 0; d < 4; ++d) {
            if (is_step_safe(d, false)) {
                int edge = s.cells[index(here)].edge[d];
                if (edge > 0)
                    return {{DIRS[d]}, 0, Mode::Portal, false};
            }
        }

        auto forced = portal_target(true);
        if (forced && forced->approach == here)
            return {{DIRS[forced->direction]}, 0, Mode::Portal, false};

        // 2. Emergency split if trapped
        if (c.can_split(c.get_length() - 2))
            return {{}, c.get_length() - 2, Mode::Split, false};

        // 3. If trapped: prefer enemy head (taking them down with us)
        for (int d = 0; d < 4; ++d) {
            if (is_step_safe(d, true))
                return {{DIRS[d]}, 0, Mode::Trapped, true};
        }

        // 4. Prefer any passable edge avoiding 180 neck reversal AND strictly avoiding friendly heads
        for (int d = 0; d < 4; ++d) {
            if (d == opp_dir || s.cells[index(here)].edge[d] < 0)
                continue;
            auto opt_dest = destination(here, d);
            if (opt_dest) {
                auto t = c.get_tile(*opt_dest);
                if (t && t->get_dragon() && t->get_dragon()->get_team() == c.get_team() && t->get_dragon()->is_head())
                    continue; // Never team-kill a friendly head!
            }
            return {{DIRS[d]}, 0, Mode::Trapped, false};
        }

        // 5. Fallback: die to self/kelp rather than taking down a friendly dragon
        for (int d = 0; d < 4; ++d) {
            auto opt_dest = destination(here, d);
            if (opt_dest) {
                auto t = c.get_tile(*opt_dest);
                if (t && t->get_dragon() && t->get_dragon()->get_team() == c.get_team() && t->get_dragon()->is_head())
                    continue;
            }
            return {{DIRS[d]}, 0, Mode::Trapped, true};
        }

        return {{c.get_dir()}, 0, Mode::Trapped, true};
    }

    void execute(const Action &a) {
        Position post = here;
        if (a.child) {
            c.do_split(a.child);
        } else {
            for (auto d : a.moves) {
                int e = s.cells[index(post)].edge[di(d)];
                auto n = destination(post, di(d));
                if (e > 0) {
                    s.pending_portal = e - 1;
                }
                if (!n)
                    break;
                auto t = c.get_tile(*n);
                if (t && t->has_pearl())
                    s.last_food = round;
                post = *n;
            }

            if (a.moves.empty()) {
                c.make_move(c.get_dir()); // Guarantees never emitting empty MOVE
            } else if (a.moves.size() == 1) {
                c.make_move(a.moves.front());
            } else {
                c.make_moves(a.moves);
            }
        }

        if (s.alpha && c.get_id() < 4096 && !a.intentional_death && s.pending_portal < 0) {
            std::uint32_t msg = 0xA7000000u | (static_cast<std::uint32_t>(c.get_id()) << 12) |
                                (static_cast<std::uint32_t>(post.x) << 6) |
                                static_cast<std::uint32_t>(post.y);
            c.send_sonar(msg);
        }

        c.set_indicator_string(std::string(s.alpha ? "Alpha:" : "Kamikaze:") + name(a.mode));
    }
};

} // namespace bot

// ============================================================================
// Main Loop
// ============================================================================

int main() {
    try {
        auto [ct, game] = unswbc::init();
        // Allocate Brain on heap to avoid WebAssembly 64KB stack overflow
        auto brain = std::make_unique<bot::Brain>(ct, game);

        while (unswbc::update(ct, game)) {
            try {
                auto action = brain->decide();
                brain->execute(action);
            } catch (...) {
                // Fail-safe: ensure a valid move is always issued before end_turn
                ct.make_move(ct.get_dir());
            }
            unswbc::end_turn();
        }
    } catch (...) {
        return 0;
    }
    return 0;
}
